import pydiffvg
from pwarp.core.arap_torch import StepOne_torch, StepTwo_torch
from pwarp.core import ops_torch


def arap_torch(vertices, faces, control_pts, shifted_locations):
    # step one
    edges = ops_torch.get_edges(len(faces), faces)
    gi, g_product = StepOne_torch.compute_g_matrix(vertices, edges, faces)
    h = StepOne_torch.compute_h_matrix(edges, g_product, gi, vertices)
    a1_matrix = StepOne_torch.compute_A_matrix(edges, vertices, gi, h, control_pts)

    # step two
    args = edges, vertices, gi, h, control_pts, shifted_locations, a1_matrix
    new_vertices, _, _ = StepOne_torch.compute_v_prime(*args)
    t_matrix = StepTwo_torch.compute_t_matrix(edges, g_product, gi, new_vertices)
    new_vertices = StepTwo_torch.compute_v_2prime(edges, vertices, t_matrix, control_pts, shifted_locations)
    return new_vertices


def warp_svg(shapes, faces, face_index, bary_coord, vertices, cum_sizes):
    """
    vertices: new positions of mesh vertices
    """
    index = faces[face_index]
    new_pts = (vertices[index] * bary_coord[:, :, None]).sum(dim=1).float()

    new_shapes = [pydiffvg.Path(
        num_control_points=shape.num_control_points,
        points=new_pts[cum_sizes[i]:cum_sizes[i+1]],
        is_closed=shape.is_closed,
    ) for i, shape in enumerate(shapes)]
    return new_shapes
