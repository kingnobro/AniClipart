import os
import numpy as np
import torch
from svglib.svg import SVG

targets = [
    ''
]

svg_template = '''<?xml version="1.0" encoding="utf-8"?>
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 width="512px" height="512px" viewBox="0 0 512 512" style="enable-background:new 0 0 512 512;" xml:space="preserve">

{}
</svg>
'''
circle_template = '<circle fill="red" cx="{}" cy="{}" r="3.2"/>'


def write_svg(circles, output_path):
    svg = svg_template.format('\n'.join(circles))
    with open(output_path, "w") as f:
        f.write(svg)

def save_skeleton(skeleton, output_path):
    with open(output_path, "w") as f:
        for sk in skeleton:
            f.write("{} {}\n".format(sk[0], sk[1]))

skeleton_path = 'svg_input/{}/{}_skeleton.txt'
keypoint_path = 'svg_input/{}/{}_keypoint.svg'

for target in targets:
    ske_path = skeleton_path.format(target, target)
    key_path = keypoint_path.format(target, target)

    control_pts = SVG.load_svg(key_path)
    control_pts = np.array([c.center.pos for c in control_pts.svg_path_groups])
    if len(control_pts) == 17:
        face_pts = control_pts[:5].mean(axis=0, keepdims=True)
        control_pts = np.concatenate([face_pts, control_pts[5:]], axis=0)
        circles = [circle_template.format(*c) for c in control_pts]
        write_svg(circles, key_path)

    with open(ske_path, 'r') as f:
        skeleton = f.readlines()
    skeleton = [s.strip().split(' ') for s in skeleton]
    skeleton = torch.tensor([[int(s[0]), int(s[1])] for s in skeleton])
    if len(skeleton) == 19:
        skeleton = skeleton[:-7] - 4
        # save
        skeleton = torch.cat([skeleton, torch.tensor([[0, 1], [0, 2]])], dim=0)
        save_skeleton(skeleton, ske_path)
