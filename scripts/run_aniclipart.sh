gpu_id=0

experiment_name="aniclipart"
targets=(
    "man_dance"
    "man_dive"
    "man_jump"

    "woman_dance3"
    "woman_wave"
    "woman_exercise"
)

for target in "${targets[@]}"; do
    output_path="output_videos"
    output_folder="${target}/${experiment_name}"
    echo "==== target: $target ===="
    echo "output folder: $output_folder"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python animate_clipart.py \
        --target "svg_input/${target}/${target}" \
        --output_path "$output_path" \
        --output_folder "$output_folder" \
        --optim_bezier_points \
        --bezier_radius 0.01 \
        --augment_frames \
        --lr_bezier 0.005 \
        --num_iter 500 \
        --num_frames 24 \
        --inter_dim 128 \
        --loop_num 2 \
        --guidance_scale 50 \
        --opt_bezier_points_with_mlp \
        --normalize_input \
        --opt_with_skeleton \
        --skeleton_weight 25 \
        --fix_start_points \
        --arap_weight 3000 \
        --max_tri_area 30 \
        --min_tri_degree 20 \
        --need_subdivide
done