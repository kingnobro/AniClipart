# AniClipart: Clipart Animation with Text-to-Video Priors

# Setup
```
git clone https://github.com/kingnobro/AniClipart.git
cd AniClipart
```

## Environment
To set up our environment, please run:
```
conda env create -f environment.yml
```
Next, you need to install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

## Run
Single-layer animation:
```
bash scripts/run_aniclipart.sh
```
Multi-layer animation:
```
bash scripts/run_layer_aniclipart.sh
```


## Keypoint Detection
For humans, we use [UniPose](https://github.com/IDEA-Research/UniPose?tab=readme-ov-file). Take a look at our example SVG input. Specifically, we merge 5 points on face (`tools.merge_unipose_ske.py`) due to the limitations of mesh-based algorithms in altering emotions, alongside the video diffusion model's inability to precisely direct facial expressions.

For broader categories, first install scikit-geometry:
```
conda install -c conda-forge scikit-geometry
```

Then put your SVG files under `svg_input`. For example, if your download SVG from the Internet and its name is `cat`, then you create the folder `svg_input/cat` and there is a file `cat.svg` in this folder.

Then, modify the `target` in `preprocess/keypoint_detection.py` and run:
```
python -m preprocess.keypoint_detection
```
You can adjust `epsilon`, `max_iter` and `factor` to adjust the complexity of the skeleton.

## SVG Preprocess
For SVG downloaded from the Internet, there may exist complex grammars.

For a file `cat_input.svg`, we first use [picosvg](https://github.com/googlefonts/picosvg) to remove grammars like `group` and `transform`:
```
picosvg cat_input.svg > cat.svg
```
Then you modify the SVG to `256x256` by running:
```
python -m preprocess.svg_resize 
```

## More Tips
1. Similar to [LiveSketch](https://github.com/yael-vinker/live_sketch), we can also use an MLP to optimize the shape of bezier by passing `opt_bezier_points_with_mlp`. This is the defaul setting in `run_clipart.sh`.
2. If you do not use an MLP, set the `lr_bezier` to `0.5`.
3. You can adjust `loop_num` to `0` to remove looping animation setting.
4. We use `fix_start_points` to freeze the initial pose.
5. Adjust `num_iter` to increase/decrease the range of motions.
6. You can adjust `max_tri_area` to change the number of triangles.
7. The `need_subdivide` option segments the SVG path. Take a look at [VectorTalker](https://arxiv.org/abs/2312.11568) (Section 3.2).