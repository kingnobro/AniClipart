import math
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg

import utils.util as util
import json

from utils.mesh_util import (
    silhouette,
    find_contour_point,
    triangulate,
    dilate,
    all_inside_contour,
    prepare_barycentric_coord,
    add_mesh_to_svg,
)
from svglib.svg import SVG
from svglib.geom import Bbox
import matplotlib.pyplot as plt

from pwarp.core.arap_torch import StepOne_torch, StepTwo_torch
from pwarp.core import ops_torch
from utils.arap_util import warp_svg


class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 svg_path: str,
                 num_frames: int,
                 device,
                 path_to_trained_mlp=None,
                 inference=False):
        super(Painter, self).__init__()
        self.svg_path = svg_path
        self.num_frames = num_frames
        self.device = device
        self.optim_bezier_points = args.optim_bezier_points
        self.opt_bezier_points_with_mlp = args.opt_bezier_points_with_mlp
        self.fix_start_points = args.fix_start_points
        self.render = pydiffvg.RenderFunction.apply
        self.normalize_input = args.normalize_input
        self.arap_weight = args.arap_weight
        self.opt_with_layered_arap = args.opt_with_layered_arap and osp.exists(f'{self.svg_path}_layer.json')
        self.loop_num = args.loop_num

        if self.optim_bezier_points:
            if self.opt_with_layered_arap:
                self.init_layered_mesh(cfg=args)
            else:
                self.init_mesh(cfg=args)
        
        if self.opt_bezier_points_with_mlp:
            self.points_bezier_mlp_input_ = self.point_bezier_mlp_input.float().unsqueeze(0).to(device)

            self.points_per_frame = 1  # FIXME
            self.mlp_points = PointMLP(input_dim=torch.numel(self.points_bezier_mlp_input_),
                                        inter_dim=args.inter_dim,
                                        num_frames=num_frames,
                                        device=device,
                                        inference=inference).to(device)
            
            if path_to_trained_mlp:
                print(f"Loading MLP from {path_to_trained_mlp}")
                self.mlp_points.load_state_dict(torch.load(path_to_trained_mlp))
                self.mlp_points.eval()

            # Init the weights of LayerNorm for global translation MLP if needed.
            if args.translation_layer_norm_weight:
                self.init_translation_norm(args.translation_layer_norm_weight)

    def init_mesh(self, cfg):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the control points of bezier paths
        """
        parameters = edict()
        parameters.point_bezier = []

        svg_path = f'{self.svg_path}_scaled.svg'
        svg_keypts_path = f'{self.svg_path}_keypoint_scaled.svg'
        src = SVG.load_svg(svg_path)
        control_pts = SVG.load_svg(svg_keypts_path)
        control_pts = np.array([c.center.pos for c in control_pts.svg_path_groups])
        self.control_pts = control_pts
        self.num_control_pts = len(control_pts)

        # init the canvas_width, canvas_height
        width = int(src.viewbox.wh.x)
        height = int(src.viewbox.wh.y)
        self.canvas_width = cfg.width = width
        self.canvas_height = cfg.height = height

        # find contour points, num is controlled by cfg.boundary_simplify_level
        contour_pts = get_contour(src, cfg)

        # dilate contour to include all pts
        contour_pts = dilate_contour(contour_pts, src)

        # prepare segments for CDT
        segments = np.array([(i, (i + 1) % len(contour_pts)) for i in range(len(contour_pts))])
        control_pts_index = np.arange(len(contour_pts), len(contour_pts) + len(control_pts))
        all_pts = np.concatenate([contour_pts, control_pts], axis=0)

        mesh = triangulate(cfg, all_pts, segments)
        plt.savefig(osp.join(cfg.mesh_dir, 'mesh.png'))
        vertices, triangles = mesh['vertices'], mesh['triangles']

        # add mesh to source for visualization
        src_mesh = add_mesh_to_svg(src.copy(), mesh)
        src_mesh.save_svg(osp.join(cfg.mesh_dir, 'source_mesh.svg'))

        src.drop_z()
        src.filter_consecutives()
        src.save_svg(osp.join(cfg.svg_dir, 'init.svg'))
        if cfg.need_subdivide:
            print('start subdivide...')
            edges = ops_torch.get_edges(len(triangles), triangles)
            src = src.subdivide(edges, vertices)
            svg_path = osp.join(cfg.svg_dir, 'init_subdiv.svg')  # update svg_path
            src.save_svg(svg_path)
            print('end subdivide...')

        # barycentric coordinate
        _, _, src_shapes, src_shape_groups = pydiffvg.svg_to_scene(svg_path)  # preprocessing done, so just load the same svg
        face_index, bary_coord = prepare_barycentric_coord(src_shapes, vertices, triangles)
        cum_sizes =  np.cumsum([shape.points.shape[0] for shape in src_shapes])
        cum_sizes = np.concatenate([[0], cum_sizes])  # used in warp_svg
        print('bary coord computed')

        # prepare ARAP
        vertices = torch.from_numpy(vertices.astype(np.float32))
        faces = torch.from_numpy(triangles.astype(np.int64))
        edges = ops_torch.get_edges(len(faces), faces)
        gi, g_product = StepOne_torch.compute_g_matrix(vertices, edges, faces)
        h = StepOne_torch.compute_h_matrix(edges, g_product, gi, vertices)
        a1_matrix = StepOne_torch.compute_A_matrix(edges, vertices, gi, h, control_pts_index, self.arap_weight)
        self.vertices = vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.edges = edges.to(self.device)
        self.gi, self.g_product = gi.to(self.device), g_product.to(self.device)
        self.h = h.to(self.device)
        self.a1_matrix = a1_matrix.to(self.device)
        self.bary_coord = bary_coord.to(self.device)
        print('ARAP prepared')

        self.control_pts_index = control_pts_index
        self.face_index = face_index
        self.src_shapes = src_shapes
        self.src_shape_groups = src_shape_groups
        self.cum_sizes = cum_sizes

        # init bezier path
        bezier_shapes, bezier_shape_groups = init_bezier_with_start_point(control_pts, width, height, cfg.bezier_radius, self.device)
        pydiffvg.save_svg(osp.join(cfg.bezier_dir, 'init_bezier.svg'), width, height, bezier_shapes, bezier_shape_groups)

        for path in bezier_shapes:
            if self.optim_bezier_points and not self.opt_bezier_points_with_mlp:
                path.points.requires_grad = True
            parameters.point_bezier.append(path.points)

        self.bezier_shapes = bezier_shapes
        self.bezier_shape_groups = bezier_shape_groups

        tensor_point_bezier_init = [torch.cat([path.points]) for path in bezier_shapes]
        self.point_bezier_mlp_input = torch.cat(tensor_point_bezier_init)  # [4*num_control_pts, 2]
        self.parameters_ = parameters

    def init_layered_mesh(self, cfg):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the control points of bezier paths
        """
        parameters = edict()
        parameters.point_bezier = []

        svg_path = f'{self.svg_path}_scaled.svg'
        svg_keypts_path = f'{self.svg_path}_keypoint_scaled.svg'
        src = SVG.load_svg(svg_path)
        control_pts = SVG.load_svg(svg_keypts_path)
        control_pts = np.array([c.center.pos for c in control_pts.svg_path_groups])
        self.control_pts = control_pts
        self.num_control_pts = len(control_pts)

        # init the canvas_width, canvas_height
        width = int(src.viewbox.wh.x)
        height = int(src.viewbox.wh.y)
        self.canvas_width = cfg.width = width
        self.canvas_height = cfg.height = height

        # layered-ARAP
        with open(f'{self.svg_path}_layer.json', 'r') as f:
            layers = json.load(f)
            self.num_of_layers = len(layers)

        # for arap
        self.vertices = []
        self.faces = []
        self.edges = []
        self.gi = []
        self.g_product = []
        self.h = []
        self.a1_matrix = []
        self.bary_coord = []
        self.control_pts_index = []
        self.face_index = []
        self.src_shapes_layer = []
        self.cum_sizes = []

        # for layered-arap
        self.control_index_layer = []
        self.path_index_layer = []

        # in order to restore the entire svg
        entire_svg = src.copy()

        for i, layer in enumerate(layers):
            path_index_layer = [index - 1 for index in layer['path_index']]
            control_index_layer = [index - 1 for index in layer['control_index']]
            self.path_index_layer.extend(path_index_layer)
            self.control_index_layer.append(np.array(control_index_layer))
            print('========== layer', i, '==========')
            print('path idx', path_index_layer)
            print('ctrl idx', control_index_layer)

            # create individual layer svg
            control_pts_layer = control_pts[control_index_layer]
            src_layer = SVG([src.svg_path_groups[index].copy() for index in path_index_layer], src.viewbox)

            # find contour points, num is controlled by cfg.boundary_simplify_level
            contour_pts = get_contour(src_layer, cfg)

            # dilate contour to include all pts
            contour_pts = dilate_contour(contour_pts, src_layer)

            # prepare segments for CDT
            segments = np.array([(i, (i + 1) % len(contour_pts)) for i in range(len(contour_pts))])
            control_pts_index = np.arange(len(contour_pts), len(contour_pts) + len(control_pts_layer))
            all_pts = np.concatenate([contour_pts, control_pts_layer], axis=0)

            mesh = triangulate(cfg, all_pts, segments)
            plt.savefig(osp.join(cfg.mesh_dir, f'layer{i}_mesh.png'))
            vertices, triangles = mesh['vertices'], mesh['triangles']

            # add mesh to source for visualization
            src_mesh = add_mesh_to_svg(src_layer.copy(), mesh)
            src_mesh.save_svg(osp.join(cfg.mesh_dir, f'layer{i}_mesh.svg'))

            src_layer.drop_z()
            src_layer.filter_consecutives()
            src_layer.save_svg(osp.join(cfg.svg_dir, f'layer{i}_init.svg'))
            if cfg.need_subdivide:
                print('start subdivide...')
                edges = ops_torch.get_edges(len(triangles), triangles)
                src_layer = src_layer.subdivide(edges, vertices)
                svg_path = osp.join(cfg.svg_dir, f'layer{i}_init_subdiv.svg')  # update svg_path
                src_layer.save_svg(svg_path)
                print('end subdivide...')

            for j, path_index in enumerate(path_index_layer):
                entire_svg.svg_path_groups[path_index] = src_layer.svg_path_groups[j].copy()

            # barycentric coordinate
            _, _, src_shapes, _ = pydiffvg.svg_to_scene(svg_path)  # preprocessing done, so just load the same svg
            face_index, bary_coord = prepare_barycentric_coord(src_shapes, vertices, triangles)
            cum_sizes = np.cumsum([shape.points.shape[0] for shape in src_shapes])
            cum_sizes = np.concatenate([[0], cum_sizes])  # used in warp_svg
            print('bary coord computed')

            # prepare ARAP
            vertices = torch.from_numpy(vertices.astype(np.float32))
            faces = torch.from_numpy(triangles.astype(np.int64))
            edges = ops_torch.get_edges(len(faces), faces)
            gi, g_product = StepOne_torch.compute_g_matrix(vertices, edges, faces)
            h = StepOne_torch.compute_h_matrix(edges, g_product, gi, vertices)
            a1_matrix = StepOne_torch.compute_A_matrix(edges, vertices, gi, h, control_pts_index, self.arap_weight)

            self.vertices.append(vertices.to(self.device))
            self.faces.append(faces.to(self.device))
            self.edges.append(edges.to(self.device))
            self.gi.append(gi.to(self.device))
            self.g_product.append(g_product.to(self.device))
            self.h.append(h.to(self.device))
            self.a1_matrix.append(a1_matrix.to(self.device))
            self.bary_coord.append(bary_coord.to(self.device))
            self.control_pts_index.append(control_pts_index)
            self.face_index.append(face_index)
            self.src_shapes_layer.append(src_shapes)
            self.cum_sizes.append(cum_sizes)

        self.path_index_layer = np.array(self.path_index_layer).flatten()
        self.path_index_layer_sorted = self.path_index_layer.argsort()

        # save entire svg
        svg_path = osp.join(cfg.svg_dir, f'entire_init_subdiv.svg')  # update svg_path
        entire_svg.save_svg(svg_path)
        _, _, _, self.src_shape_groups = pydiffvg.svg_to_scene(svg_path)

        # init bezier path
        bezier_shapes, bezier_shape_groups = init_bezier_with_start_point(control_pts, width, height, cfg.bezier_radius, self.device)
        pydiffvg.save_svg(osp.join(cfg.bezier_dir, 'init_bezier.svg'), width, height, bezier_shapes, bezier_shape_groups)

        for path in bezier_shapes:
            if self.optim_bezier_points and not self.opt_bezier_points_with_mlp:
                path.points.requires_grad = True
            parameters.point_bezier.append(path.points)

        self.bezier_shapes = bezier_shapes
        self.bezier_shape_groups = bezier_shape_groups

        tensor_point_bezier_init = [torch.cat([path.points]) for path in bezier_shapes]
        self.point_bezier_mlp_input = torch.cat(tensor_point_bezier_init)  # [4*num_control_pts, 2]
        self.parameters_ = parameters

    def render_frames_to_tensor_direct_optim_bezier(self, point_bezier):
        # point_bezier: List[Tensor], each tensor is [4, 2]
        frames_init, frames_svg, points_init_frame = [], [], []

        shifted_locations = []  # compute points on bezier curves
        for t in self.sample_on_bezier_path(self.loop_num):
            loc = torch.stack([cubic_bezier(p, t) for p in point_bezier])
            shifted_locations.append(loc.unsqueeze(0))
        shifted_locations = torch.cat(shifted_locations, dim=0)  # [frame_num, num_bezier, 2]

        # ARAP
        new_vertices, _, _ = StepOne_torch.compute_v_prime(self.edges, self.vertices, self.gi, self.h, self.control_pts_index, shifted_locations, self.a1_matrix, device=self.device, weight=self.arap_weight)
        t_matrix = StepTwo_torch.compute_t_matrix(self.edges, self.g_product, self.gi, new_vertices, device=self.device)
        new_vertices = StepTwo_torch.compute_v_2prime(self.edges, self.vertices, t_matrix, self.control_pts_index, shifted_locations, device=self.device, weight=self.arap_weight)
        
        # warp svg based on the updated mesh
        for vs in new_vertices:
            new_shapes = warp_svg(self.src_shapes, self.faces, self.face_index, self.bary_coord, vs, self.cum_sizes)
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes, self.src_shape_groups)

            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]

            frames_init.append(cur_im)
            frames_svg.append((new_shapes, self.src_shape_groups))
        
        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init = frames_init + frames_init[::-1]
            frames_svg = frames_svg + frames_svg[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init = frames_init + frames_init
                frames_svg = frames_svg + frames_svg

        return torch.stack(frames_init), frames_svg, points_init_frame, shifted_locations, point_bezier
    
    def render_frames_to_tensor_direct_optim_bezier_layered(self, point_bezier):
        # point_bezier: List[Tensor], each tensor is [4, 2]
        frames_init, frames_svg, points_init_frame = [], [], []

        shifted_locations = []  # compute points on bezier curves
        ts = self.sample_on_bezier_path(self.loop_num)
        for t in ts:
            loc = torch.stack([cubic_bezier(p, t) for p in point_bezier])
            shifted_locations.append(loc.unsqueeze(0))
        shifted_locations = torch.cat(shifted_locations, dim=0)  # [frame_num, num_bezier, 2]

        new_vertices_layer = [[] for _ in range(len(ts))]
        for i in range(self.num_of_layers):
            # arap
            vertices = self.vertices[i]
            edges = self.edges[i]
            faces = self.faces[i]
            gi = self.gi[i]
            g_product = self.g_product[i]
            h = self.h[i]
            a1_matrix = self.a1_matrix[i]
            bary_coord = self.bary_coord[i]
            control_pts_index = self.control_pts_index[i]
            face_index = self.face_index[i]
            src_shapes = self.src_shapes_layer[i]
            cum_sizes = self.cum_sizes[i]

            # layered-arap
            control_index_layer = self.control_index_layer[i]
            shifted_locations_layer = shifted_locations[:, control_index_layer, :]
            
            new_vertices, _, _ = StepOne_torch.compute_v_prime(edges, vertices, gi, h, control_pts_index, shifted_locations_layer, a1_matrix, device=self.device, weight=self.arap_weight)
            t_matrix = StepTwo_torch.compute_t_matrix(edges, g_product, gi, new_vertices, device=self.device)
            new_vertices = StepTwo_torch.compute_v_2prime(edges, vertices, t_matrix, control_pts_index, shifted_locations_layer, device=self.device, weight=self.arap_weight)
            
            # warp svg based on the updated mesh
            for j, vs in enumerate(new_vertices):
                new_vertices_layer[j].extend(warp_svg(src_shapes, faces, face_index, bary_coord, vs, cum_sizes))
        
        # construct entire svg
        for new_vertices in new_vertices_layer:  # num of ts
            # sort new_shapes according to path_index_layer
            new_shapes = [new_vertices[i] for i in self.path_index_layer_sorted]

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes, self.src_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]

            frames_init.append(cur_im)
            frames_svg.append((new_shapes, self.src_shape_groups))
        
        # motion repeat
        if self.loop_num > 0:
            # one loop
            frames_init = frames_init + frames_init[::-1]
            frames_svg = frames_svg + frames_svg[::-1]
            if self.loop_num > 1:
                # two loops
                frames_init = frames_init + frames_init
                frames_svg = frames_svg + frames_svg

        return torch.stack(frames_init), frames_svg, points_init_frame, shifted_locations, point_bezier
    
    def render_frames_to_tensor_mlp_bezier(self):
        frame_input = self.points_bezier_mlp_input_
        if self.normalize_input:
            frame_input = util.normalize_tensor(frame_input)  # [0, 1]
        # predict the delta of control points of all bezier paths
        delta_prediction = self.mlp_points(frame_input)  # [4 * num_control_pts, 2]

        # add predicted delta to the original bezier shapes
        point_bezier = []
        for i in range(self.num_control_pts):
            updated_points = self.point_bezier_mlp_input[i * 4 : (i + 1) * 4] + delta_prediction[i * 4 : (i + 1) * 4]
            if self.fix_start_points:
                updated_points[0] = self.point_bezier_mlp_input[i * 4]
            point_bezier.append(updated_points)
            # update shapes for visualization
            self.bezier_shapes[i].points = updated_points.detach()

        if self.opt_with_layered_arap:
            return self.render_frames_to_tensor_direct_optim_bezier_layered(point_bezier)
        return self.render_frames_to_tensor_direct_optim_bezier(point_bezier)

    def render_frames_to_tensor_with_bezier(self, mlp=True):
        if self.opt_bezier_points_with_mlp and mlp:
            return self.render_frames_to_tensor_mlp_bezier()
        else:
            if self.opt_with_layered_arap:
                return self.render_frames_to_tensor_direct_optim_bezier_layered(self.parameters_["point_bezier"])
            return self.render_frames_to_tensor_direct_optim_bezier(self.parameters_["point_bezier"])
    
    def get_bezier_params(self):
        if self.opt_bezier_points_with_mlp:
            return self.mlp_points.get_points_params()
        return self.parameters_["point_bezier"]

    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if self.opt_bezier_points_with_mlp:
            torch.save(self.mlp_points.state_dict(), f"{output_path}/model.pt")
            print(f"Model saved to {output_path}/model.pt")

    def init_translation_norm(self, translation_layer_norm_weight):
        print(f"Initializing translation layerNorm to {translation_layer_norm_weight}")
        for child in self.mlp_points.frames_rigid_translation.children():
            if isinstance(child, nn.LayerNorm):
                with torch.no_grad():
                    child.weight *= translation_layer_norm_weight

    def sample_on_bezier_path(self, loop_num):
        segment_len = self.num_frames if loop_num == 0 else self.num_frames // (loop_num * 2)
        ts = torch.linspace(0, 1, segment_len)
        return ts


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        # x = x + 
        return self.dropout(self.pe[:x.size(0), :])


class PointModel(nn.Module):

    def __init__(self, input_dim, inter_dim, num_frames, device, inference=False):

        super().__init__()
        self.num_frames = num_frames
        self.inter_dim = inter_dim
        self.input_dim = input_dim
        self.embed_dim = inter_dim
        self.inference = inference

        self.project_points = nn.Sequential(nn.Linear(2, inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(inter_dim, inter_dim))

        self.embedding = nn.Embedding(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=self.embed_dim, max_len=input_dim)
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)


    def get_position_encoding_representation(self, init_points):
        # input dim: init_points [num_frames * points_per_frame, 2], for ballerina [832,2] = [16*52, 2]
        # the input are the points of the given initial frame (user's drawing)
        # note that we calculate the point's distance from the object's center, and operate on this distance
        emb_xy = self.project_points(init_points)  # output shape: [1,num_frames * points_per_frame,128] -> [1,832,128]
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim)  # inds dim is N*K, embed dim is [N*K, 128]
        pos = self.pos_encoder(embed.unsqueeze(1)).permute(1, 0, 2)  # [1, N*K, 128]
        init_points_pos_enc = emb_xy + pos  # [1, N*K, 128]
        return init_points_pos_enc

    def forward(self, init_points):
        raise NotImplementedError("PointModel is an abstract class. Please inherit from it and implement a forward function.")

    def get_shared_params(self):
        project_points_p = list(self.project_points.parameters())
        embedding_p = list(self.embedding.parameters())
        pos_encoder_p = list(self.pos_encoder.parameters())

        return project_points_p + embedding_p + pos_encoder_p
        
    def get_points_params(self):
        shared_params = self.get_shared_params()
        project_xy_p = list(self.project_xy.parameters())
        model_p = list(self.model.parameters())
        last_lin = list(self.last_linear_layer.parameters())
        return shared_params + project_xy_p + model_p + last_lin
        

class PointMLP(PointModel):
    def __init__(self, input_dim, inter_dim, num_frames, device, inference):

        super().__init__(input_dim, inter_dim, num_frames, device, inference)

        self.project_xy = nn.Sequential(nn.Flatten(),
                                        nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                        nn.LayerNorm(inter_dim),
                                        nn.LeakyReLU())

        self.model = nn.Sequential(
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
        )

        self.last_linear_layer = nn.Linear(inter_dim, input_dim)

    def forward(self, init_points):
        init_points_pos_enc = self.get_position_encoding_representation(init_points)

        project_xy = self.project_xy(init_points_pos_enc)  # Flatten, output is [1, 128]
        delta = self.model(project_xy)  # [1,128]
        delta_xy = self.last_linear_layer(delta).reshape(init_points.shape)  # [1,128] -> [1, N*K, 2]

        return delta_xy.squeeze(0)

class PainterOptimizer:
    def __init__(self, args, painter):
        self.painter = painter
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.lr_bezier = args.lr_bezier
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.optim_bezier_points = args.optim_bezier_points
        self.init_optimizers()

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def init_optimizers(self):
        if self.optim_bezier_points:
            bezier_delta_params = self.painter.get_bezier_params()
            self.bezier_delta_optimizer = torch.optim.Adam(bezier_delta_params, lr=self.lr_bezier,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_bezier = LambdaLR(self.bezier_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        if self.optim_bezier_points:
            self.scheduler_bezier.step()

    def zero_grad_(self):
        if self.optim_bezier_points:
            self.bezier_delta_optimizer.zero_grad()

    def step_(self):
        if self.optim_bezier_points:
            self.bezier_delta_optimizer.step()
        if self.painter.fix_start_points and not self.painter.opt_bezier_points_with_mlp:
            with torch.no_grad():
                for i in range(self.painter.num_control_pts):
                    self.painter.parameters_["point_bezier"][i][0] = self.painter.point_bezier_mlp_input[i * 4]

    def get_lr(self, optim="points"):
        if optim == "bezier_points" and self.optim_bezier_points:
            return self.bezier_delta_optimizer.param_groups[0]['lr']
        else:
            return None


def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)
    return center, all_points


def get_deltas(all_points, center, device):
    deltas_from_center = []
    for points in all_points:
        deltas = (points - center).to(device)
        deltas_from_center.append(deltas)
    return deltas_from_center


def get_contour(svg, cfg, render_size=512):
    svg = svg.copy().normalize(Bbox(render_size, render_size))  # high resolution can produce accurate contour
    src_png = svg.draw(return_png=True, do_display=False)
    s = silhouette(src_png)  # black & white
    s_cont, contour_pts = find_contour_point(s, epsilon=cfg.boundary_simplify_level)
    s.save(osp.join(cfg.mesh_dir, 'silhouette.png'))
    s_cont.save(osp.join(cfg.mesh_dir, 'silhouette_contour.png'))
    contour_pts = contour_pts.astype(np.float32) / (render_size / cfg.render_size_h)
    return contour_pts


def dilate_contour(contour_pts, svg):
    # dilate contour to include all pts of svg
    step = 0.1
    total_step = step
    svg_pts = svg.to_points()
    while not all_inside_contour(svg_pts, contour_pts):
        contour_pts = dilate(contour_pts, step)
        total_step += step
    print('contour expansion:', total_step)
    return contour_pts


def init_bezier_with_start_point(start_points, W, H, radius=1, device='cpu'):
    def perturb_point(p, radius=1):
        return [p[0] + radius * (random.random() - 0.5),
                p[1] + radius * (random.random() - 0.5)]

    shapes = []
    shape_groups = []
    for p0 in start_points:
        p1 = perturb_point(p0, radius)
        p2 = perturb_point(p1, radius)
        p3 = perturb_point(p2, radius)
        
        points = torch.tensor(np.array([p0, p1, p2, p3])).to(device)
        points[:, 0] = points[:, 0].clip(min=5, max=W-5)
        points[:, 1] = points[:, 1].clip(min=5, max=H-5)

        path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                             points=points,
                             stroke_width=torch.tensor(0.5),
                             is_closed=False)
        shapes.append(path)  # must `append` before creating path_group
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=None,
                                         stroke_color=torch.tensor([1.0, 0.0, 0.0, 1]))
        shape_groups.append(path_group)

    return shapes, shape_groups


def cubic_bezier(P, t):
    return (1.0-t)**3*P[0] + 3*(1.0-t)**2*t*P[1] + 3*(1.0-t)*t**2*P[2] + t**3*P[3]
