import os
import os.path as osp
from svglib.svg import SVG
from svglib.svg_command import SVGCommandLine
from svglib.svg_path import SVGPath
from svglib.svg_primitive import SVGPathGroup, SVGCircle, Radius
from svglib.geom import Point
import skgeom as sg
import numpy as np
from skgeom.draw import draw
import matplotlib.pyplot as plt
from copy import deepcopy
from utils.mesh_util import silhouette, find_contour_point


def get_contour(svg, target, epsilon=5, output_dir='./tmp'):
    src_png = svg.draw(return_png=True, do_display=False)  # Bbox 512
    s = silhouette(src_png)  # black & white
    s_cont, contour_pts = find_contour_point(s, epsilon=epsilon)
    # s.save(osp.join(output_dir, f'{target}_silhouette.png'))
    # s_cont.save(osp.join(output_dir, f'{target}_silhouette_contour.png'))
    return contour_pts.astype(np.float32)


def get_straight_skeleton(polygon, skeleton):

    def Point2np(p):
        return np.array([p.x(), p.y()], dtype=np.float32)
    
    contour_vertices = [v for v in polygon.vertices]
    edges = set()
    keypoint = {}

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1, p2 = h.vertex.point, h.opposite.vertex.point
            if p1 not in contour_vertices and p2 not in contour_vertices:  # remove perimeter edges
                id1, id2 = sorted((h.vertex.id, h.opposite.vertex.id))
                edges.add((id1, id2))
                keypoint[id1] = Point2np(p1)
                keypoint[id2] = Point2np(p2)
    
    # Create a mapping from old ids to new ids
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(keypoint))}
    
    # Update edges and keypoint with new ids
    edges = [(id_mapping[id1], id_mapping[id2]) for id1, id2 in edges]
    keypoint = {id_mapping[id]: p for id, p in keypoint.items()}

    return edges, keypoint


def collapse_edge(edges, keypoint, edge):

    def adj_edges(id):
        return [e for e in edges if id in e]
    
    id0, id1 = edge
    n0 = keypoint[id0]
    n1 = keypoint[id1]

    # find position for new node
    n0_adj_edges = adj_edges(id0)
    n1_adj_edges = adj_edges(id1)

    retain_id = False
    if len(n0_adj_edges) == 2 and len(n1_adj_edges) == 1:  # use terminal node's position
        new_pos = n1
    elif len(n0_adj_edges) == 1 and len(n1_adj_edges) == 2:  # use terminal node's position
        new_pos = n0
    elif len(n0_adj_edges) == 2 and len(n1_adj_edges) == 2:  # use both normal_nodes' avg
        new_pos = (n0 + n1) * 0.5
    elif len(n0_adj_edges) >= 3 and len(n1_adj_edges) <= 2:  # use junction node's position
        new_pos = n0
        retain_id = True
    elif len(n0_adj_edges) <= 2 and len(n1_adj_edges) >= 3:  # use junction node's position
        new_pos = n1
        retain_id = True
    elif len(n0_adj_edges) >= 3 and len(n1_adj_edges) >= 3:  # use both junction nodes' avg
        new_pos = (n0 + n1) * 0.5
    else:
        raise ValueError("Invalid edge collapse")
    
    if not retain_id:
        new_id = max(keypoint.keys()) + 1
    elif np.array_equal(new_pos, n0):
        new_id = id0
    else:
        new_id = id1
    
    # add new edges & remove old edges
    for e in n0_adj_edges:
        if e != edge:
            new_edge = tuple(sorted((new_id, e[1] if e[0] == id0 else e[0])))
            edges.append(new_edge)
            edges.remove(e)
    for e in n1_adj_edges:
        if e != edge:
            new_edge = tuple(sorted((new_id, e[1] if e[0] == id1 else e[0])))
            edges.append(new_edge)
            edges.remove(e)
    
    # remove edges & keypoints
    edges.remove(edge)
    del keypoint[id0]
    del keypoint[id1]
    keypoint[new_id] = new_pos
    
    return edges, keypoint


def simplify_by_collapse_short_edges(edges, keypoint, max_iter=5, factor=0.5):
    for step in range(max_iter):
        print("iter", step, "edges", len(edges), "keypoint", len(keypoint))

        edges_copy = deepcopy(edges)
        edge_lengths = [np.linalg.norm(keypoint[id1] - keypoint[id2]) for id1, id2 in edges_copy]
        avg_edge_length = np.mean(edge_lengths)
        thresh = factor * avg_edge_length

        need_collapse = False
        for i, edge in enumerate(edges_copy):
            if edge not in edges:
                continue

            if  edge_lengths[i] < thresh:
                edges, keypoint = collapse_edge(edges, keypoint, edge)
                need_collapse = True
        
        if not need_collapse:
            break

    # update id_mapping
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(keypoint))}
    edges = [(id_mapping[id1], id_mapping[id2]) for id1, id2 in edges]
    keypoint = {id_mapping[id]: p for id, p in keypoint.items()}

    return edges, keypoint


def draw_skeleton(svg, edges, keypoint, target, name, output_dir='./tmp'):
    new_svg = svg.copy().set_opacity(0.4)
    for edge in edges:
        a = Point(keypoint[edge[0]])
        b = Point(keypoint[edge[1]])
        l = SVGCommandLine(a, b)
        new_svg.add_path_group(SVGPathGroup(
            [SVGPath([l])],
            color='black', fill=False, stroke_width='2.0',
            # dasharray='4'
        ))

    new_svg.save_svg(f'{output_dir}/{target}_{name}.svg')


def save_skeleton(skeleton, target, name, output_dir='./tmp'):
    with open(osp.join(output_dir, f"{target}_{name}.txt"), "w") as f:
        for sk in skeleton:
            f.write("{} {}\n".format(sk[0], sk[1]))


def save_keypoint(keypoint, width, height, target, name, output_dir='./tmp'):
    svg_template = '''<?xml version="1.0" encoding="utf-8"?>
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
    width="512px" height="512px" viewBox="0 0 {} {}" style="enable-background:new 0 0 512 512;" xml:space="preserve">

{}
</svg>
    '''
    circle_template = '<circle fill="black" cx="{}" cy="{}" r="4.4"/>'

    ps = keypoint
    circles = [circle_template.format(c[0], c[1]) for c in ps]
    svg = svg_template.format(width, height, '\n'.join(circles))
    svg_path = osp.join(output_dir, f"{target}_{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)


if __name__ == '__main__':
    target = 'kpt_test'
    epsilon = 5  # more complex contour
    max_iter = 10
    factor = 0.75  # more longer edges

    svg_path = f'svg_input/{target}/{target}.svg'
    svg = SVG.load_svg(svg_path)
    width = int(svg.viewbox.wh.x)
    height = int(svg.viewbox.wh.y)

    output_dir = os.path.dirname(svg_path)

    contour_pts = get_contour(svg, target=target, epsilon=epsilon, output_dir=output_dir)
    contour_pts = contour_pts[::-1]  # counter-clockwise
    # save_keypoint(contour_pts, width, height, target, name='contour_keypoint', output_dir=output_dir)
    # contour = [[i, (i+1) % len(contour_pts)] for i in range(len(contour_pts))]
    # draw_skeleton(svg, contour, contour_pts, target=target, name='contour_skeleton', output_dir=output_dir)

    # straight skeleton
    polygon = sg.Polygon([sg.Point2(int(pt[0]), int(pt[1])) for pt in contour_pts])
    skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)
    edges, keypoint = get_straight_skeleton(polygon, skeleton)
    # draw_skeleton(svg, edges, keypoint, target=target, name='dirty', output_dir=output_dir)
    # save_skeleton(edges, target, name='dirty_skeleton', output_dir=output_dir)
    # save_keypoint([keypoint[id] for id in sorted(keypoint)], width, height, target, name='dirty_keypoint', output_dir=output_dir)

    # simplify
    edges, keypoint = simplify_by_collapse_short_edges(edges, keypoint, max_iter, factor)
    draw_skeleton(svg, edges, keypoint, target=target, name='clean', output_dir=output_dir)
    save_skeleton(edges, target, name='skeleton', output_dir=output_dir)
    save_keypoint([keypoint[id] for id in sorted(keypoint)], width, height, target, name='keypoint', output_dir=output_dir)

    print("final edges", len(edges))
    print("final keypoint", len(keypoint))
