import os
import argparse

from svglib.svg import SVG
from svglib.geom import Bbox


def process_svg(svg: SVG, scale, width, height, simplify=False, max_dist=5):
    svg.zoom(scale)
    svg.normalize(Bbox(width, height))
    if simplify:
        svg.simplify_arcs()
        svg.simplify_heuristic()
        svg.split(max_dist=max_dist)
    return svg


if __name__ == "__main__":
    targets = [
        "your_svg_name",
    ]

    for target in targets:
        parser = argparse.ArgumentParser()
        parser.add_argument("--svg_path", type=str, default=f"svg_input/{target}/{target}.svg", help="path to SVG file")
        parser.add_argument("--keypoint_path", type=str, default=f"svg_input/{target}/{target}_keypoint.svg", help="path to SVG keypoint file")
        parser.add_argument("--max_dist", type=int, default=5, help="path longer than this will be split")
        args = parser.parse_args()

        svg = SVG.load_svg(args.svg_path)
        svg.drop_z()
        svg.filter_duplicates()
        svg.filter_consecutives()
        svg.filter_empty()
        keypoint = SVG.load_svg(args.keypoint_path)

        scale, width, height = 1.0, 256, 256
        simplify = True
        svg = process_svg(svg, scale, width, height, simplify=simplify, max_dist=args.max_dist)
        keypoint = process_svg(keypoint, scale, width, height)

        svg_path = os.path.splitext(os.path.basename(args.svg_path))[0]
        svg.save_svg(f"svg_input/{svg_path}/{svg_path}_scaled.svg")
        keypoint.save_svg(f"svg_input/{svg_path}/{os.path.splitext(os.path.basename(args.keypoint_path))[0]}_scaled.svg")
