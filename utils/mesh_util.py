import numpy as np
import cv2
import triangle as tr
import matplotlib.pyplot as plt
import shapely.geometry as geo
import pydiffvg
import torch

from svglib.svg import SVG
from svglib.svg_primitive import SVGPathGroup
from svglib.svg_path import SVGPath
from svglib.svg_command import SVGCommandLine
from svglib.geom import Point, Bbox
from PIL import Image


def rectify_y(mesh: dict, height=512):
    # svg coordinate system is different from image coordinate system
    # this function will modify mesh `in-place`
    v = mesh['vertices']
    v[:, 1] = height - v[:, 1]
    return mesh


def silhouette(img):
    img.load()
    black = Image.new("RGB", img.size, (0, 0, 0))
    new_img = Image.new("RGB", img.size, (255, 255, 255))
    new_img.paste(black, mask=img.split()[3])
    return new_img


def find_contour_point(img, epsilon=3):
    # img is silhouette
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)  # reduce number of points
    cv2.drawContours(img, contour, -1, (255, 0, 0), 3)
    contour = contour.squeeze(1)
    return Image.fromarray(img), contour


def add_mesh_to_svg(svg: SVG, mesh: dict):
    mesh_svg = create_svg_mesh(mesh)
    svg.add_path_groups(mesh_svg.svg_path_groups)
    return svg


def create_svg_mesh(mesh):
    vertices = mesh['vertices']
    triangles = mesh['triangles']
    mesh_svg = SVG([], viewbox=Bbox(512))
    for tri in triangles:
        a = Point(vertices[tri[0]])
        b = Point(vertices[tri[1]])
        c = Point(vertices[tri[2]])
        l1 = SVGCommandLine(a, b)
        l2 = SVGCommandLine(b, c)
        l3 = SVGCommandLine(c, a)
        mesh_svg.add_path_group(SVGPathGroup(
            [SVGPath([l1, l2, l3])],
            color='black', fill=False, stroke_width='0.5'
        ))
    return mesh_svg


def triangulate(cfg, all_pts, segments):
    contour = dict(vertices=all_pts, segments=segments)
    flag = f'pq{cfg.min_tri_degree}a{cfg.max_tri_area}'  # https://rufat.be/triangle/API.html
    mesh = tr.triangulate(contour, flag)
    tmp_mesh = dict(vertices=mesh['vertices'].copy(), triangles=mesh['triangles'])  # avoid in-place modification
    tr.compare(plt, rectify_y(contour, cfg.height), rectify_y(tmp_mesh, cfg.height), figsize=(20, 15))
    return mesh


def dilate(vertex, distance):
    new_poly = geo.Polygon(vertex).buffer(distance, join_style=2)  # https://gis.stackexchange.com/questions/380470/what-exactly-are-cap-style-and-join-style-in-shapely-buffer-function
    return new_poly.exterior.coords[:-1]  # exclude duplicated start points


def all_inside_contour(vertex, contour):
    poly = geo.Polygon(contour)
    out_cnt = 0
    for v in vertex:
        flag = geo.Point(v).within(poly)
        out_cnt += int(not flag)
    # log(out_cnt)
    return out_cnt == 0


def get_barycentric_coord(point, vertices, faces):
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        # Compute vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        # Compute dot products
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        # Compute barycentric coordinates
        denom = d00 * d11 - d01 * d01
        b1 = (d11 * d20 - d01 * d21) / denom
        b2 = (d00 * d21 - d01 * d20) / denom
        b0 = 1.0 - b1 - b2
        # Check if point is in triangle
        if 0 <= b0 <= 1 and 0 <= b1 <= 1 and 0 <= b2 <= 1:
            return i, [b0, b1, b2]
    return None, None


def prepare_barycentric_coord(shapes: pydiffvg.shape.Path, vertices, faces):
    face_index = []
    bary_coord = []
    for shape in shapes:
        f_i, b_c = [], []
        for p in shape.points:
            index, coord = get_barycentric_coord(p, vertices, faces)
            assert index is not None and coord is not None, 'point not in mesh'
            f_i.append(index)
            b_c.append(coord)
        face_index.append(torch.tensor(f_i))
        bary_coord.append(torch.tensor(b_c))
    face_index = torch.cat(face_index)  # this will lose shape info
    bary_coord = torch.cat(bary_coord)
    return face_index, bary_coord
