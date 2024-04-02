from painter_clipart import Painter, PainterOptimizer
from losses import SDSVideoLoss, SkeletonLoss
import utils.util as util
import os
import matplotlib.pyplot as plt
import torch
import pydiffvg
from tqdm import tqdm
from pytorch_lightning import seed_everything
import argparse
import wandb
import numpy as np
from torchvision import transforms
import torchvision


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--target", type=str, default="svg_input/man_dance", help="file name of the svg to be animated")
    parser.add_argument("--caption", type=str, default="", help="Prompt for animation. verify first that this prompt works with the original text2vid model. If left empty will try to find prompt in utils.py")
    parser.add_argument("--output_path", type=str, default="output_vidoes", help="top folder name to save the results")
    parser.add_argument("--output_folder", type=str, default="horse_256", help="folder name to save the results")
    parser.add_argument("--seed", type=int, default=1000)

    # Diffusion related & Losses
    parser.add_argument("--model_name", type=str, default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--render_size_h", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--render_size_w", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--num_frames", type=int, default=24, help="should fit the default settings of the chosen video model (under 'model_name')")
    
    # SDS relted
    parser.add_argument("--sds_timestep_low", type=int, default=50) 
    parser.add_argument("--same_noise_for_frames", action="store_true", help="sample noise for one frame and repeat across all frames")
    parser.add_argument("--augment_frames", action="store_true", help="whether to randomely augment the frames to prevent adversarial results")

    # Memory saving related
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers for unet")
    parser.add_argument("--del_text_encoders", action="store_true", help="delete text encoder and tokenizer after encoding the prompts")

    # Optimization related
    parser.add_argument("--num_iter", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--optim_bezier_points", action='store_true', help="whether to optimize the bezier points")
    parser.add_argument("--opt_bezier_points_with_mlp", action='store_true', help="whether to optimize the bezier points with an MLP")
    parser.add_argument("--opt_with_skeleton", action='store_true', help="whether to optimize the bezier points with skeleton constraint")
    parser.add_argument("--opt_with_layered_arap", action='store_true', help="whether to use layered-arap")
    parser.add_argument("--fix_start_points", action='store_true', help="whether to fix the start point of bezier points")
    parser.add_argument("--loop_num", type=int, default=1, help="loop animation")

    # MLP architecture (points)
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--normalize_input", action='store_true', help="whether to normalize the input to the MLP")
    parser.add_argument("--translation_layer_norm_weight", type=int, default=0)

    # Weight
    parser.add_argument("--skeleton_weight", type=float, default=15.0, help="Scale factor for skeleton loss")

    # Learning rate related (can be simplified, taken from vectorFusion)
    parser.add_argument("--lr_init", type=float, default=0.002)
    parser.add_argument("--lr_final", type=float, default=0.0008)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--lr_delay_steps", type=float, default=100)
    parser.add_argument("--lr_bezier", type=float, default=0.1)
    parser.add_argument("--const_lr", type=int, default=0)

    # Display related
    parser.add_argument("--display_iter", type=int, default=100)
    parser.add_argument("--save_vid_iter", type=int, default=100)

    # wandb
    parser.add_argument("--report_to_wandb", action='store_true')
    parser.add_argument("--wandb_user", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--folder_as_wandb_run_name", action="store_true")

    # create mesh
    parser.add_argument("--boundary_simplify_level", type=int, default=1)
    parser.add_argument("--min_tri_degree", type=int, default=30)
    parser.add_argument("--max_tri_area", type=int, default=40)
    parser.add_argument("--arap_weight", type=float, default=3000)
    parser.add_argument("--need_subdivide", action='store_true', help="whether to make every command contained in a single triangle")

    # bezier path
    parser.add_argument("--bezier_radius", type=float, default=1.0)

    args = parser.parse_args()
    seed_everything(args.seed)

    if not args.caption:
        args.caption = util.get_clipart_caption(args.target)
        print("get caption:", args.caption)
        
    print("=" * 50)
    print("target:", args.target)
    print("caption:", args.caption)
    print("=" * 50)

    if args.folder_as_wandb_run_name:
        args.wandb_run_name = args.output_folder

    args.output_folder = f"./{args.output_path}/{args.output_folder}"
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f"{args.output_folder}/svg_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/mp4_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/mesh_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/bezier_logs", exist_ok=True)
    args.svg_dir = f"{args.output_folder}/svg_logs"
    args.mesh_dir = f"{args.output_folder}/mesh_logs"
    args.bezier_dir = f"{args.output_folder}/bezier_logs"
    
    if args.report_to_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=args.wandb_run_name, id=wandb.util.generate_id())

    import yaml
    with open(f'{args.output_folder}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    return args


def plot_video_seq(x_aug, orig_aug, cfg, step):
    pair_concat = torch.cat([orig_aug.squeeze(0).detach().cpu(), x_aug.squeeze(0).detach().cpu()])
    grid_img = torchvision.utils.make_grid(pair_concat, nrow=cfg.num_frames)
    plt.figure(figsize=(30,10))
    plt.imshow(grid_img.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"frames_iter{step}")
    plt.tight_layout()
    if cfg.report_to_wandb:
        wandb.log({"frames": wandb.Image(plt)}, step=step)


def get_augmentations():
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans


if __name__ == "__main__":
    cfg = parse_arguments()

    # Everything about rasterization and curves is defined in the Painter class
    painter = Painter(cfg, cfg.target, num_frames=cfg.num_frames, device=cfg.device)
    optimizer = PainterOptimizer(cfg, painter)
    data_augs = get_augmentations()

    # Just to test that the svg and initial frames were loaded as expected
    with torch.inference_mode():
        frames_tensor, frames_svg, points_init_frame, _, _ = painter.render_frames_to_tensor_with_bezier(mlp=cfg.opt_bezier_points_with_mlp)
    output_vid_path = f"{cfg.output_folder}/init_vid.mp4"
    util.save_mp4_from_tensor(frames_tensor, output_vid_path)

    if cfg.report_to_wandb:
        video_to_save = frames_tensor.permute(0,3,1,2).detach().cpu().numpy()
        video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
        wandb.log({"video_init": wandb.Video(video_to_save, fps=8)})
                       
    sds_loss = SDSVideoLoss(cfg, cfg.device)
    if cfg.opt_with_skeleton:
        skeleton_loss = SkeletonLoss(cfg, cfg.target, painter.control_pts, cfg.device, cfg.mesh_dir)

    orig_frames = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
    orig_frames = orig_frames.repeat(cfg.batch_size, 1, 1, 1, 1)

    sds_losses_and_opt_kwargs = []
    sds_losses_and_opt_kwargs.append((sds_loss, {}))

    t_range = tqdm(range(cfg.num_iter + 1))
    for step in t_range:
        for curr_sds_loss, opt_kwargs in sds_losses_and_opt_kwargs:
            loss_kwargs = {}
            logs = {}
            optimizer.zero_grad_()

            # Render the frames (inc. network forward pass)
            vid_tensor, frames_svg, new_points, shifted_locations, point_bezier = painter.render_frames_to_tensor_with_bezier(mlp=cfg.opt_bezier_points_with_mlp) # (K, 256, 256, 3)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
            x = x.repeat(cfg.batch_size, 1, 1, 1, 1)

            # Apply augmentations if needed
            if cfg.augment_frames:
                augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
                x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
                orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
            else:
                x_aug = x
                orig_frames_aug = orig_frames
            
            loss_sds = curr_sds_loss(x_aug, **loss_kwargs)
            loss = loss_sds

            if cfg.opt_with_skeleton:
                loss_skeleton = skeleton_loss(shifted_locations)
                loss_skeleton = cfg.skeleton_weight * loss_skeleton
                loss = loss + loss_skeleton

            t_range.set_postfix({'loss': loss.item()})
            loss.backward()

            optimizer.step_(**opt_kwargs)
            
            loss_suffix = "_global" if "skip_points" in opt_kwargs else ""
            logs.update({f"loss{loss_suffix}": loss.detach().item(), f"loss_sds{loss_suffix}": loss_sds.detach().item()})

        if not cfg.const_lr:
            optimizer.update_lr()

        logs.update({"lr_points": optimizer.get_lr("bezier_points"), "step": step})

        if cfg.report_to_wandb:
            wandb.log(logs, step=step)

        if step % cfg.save_vid_iter == 0:
            util.save_mp4_from_tensor(vid_tensor, f"{cfg.output_folder}/mp4_logs/{step}.mp4")
            util.save_vid_svg(frames_svg, f"{cfg.output_folder}/svg_logs", step, painter.canvas_width, painter.canvas_height)
            util.save_hq_video(cfg.output_folder, iter_=step, is_last_iter=(step == cfg.num_iter))
            if cfg.report_to_wandb:
                video_to_save = vid_tensor.permute(0,3,1,2).detach().cpu().numpy()
                video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
                wandb.log({"video": wandb.Video(video_to_save, fps=8)}, step=step)
                plot_video_seq(x_aug, orig_frames_aug, cfg, step)
            
            if step > 0:
                painter.log_state(f"{cfg.output_folder}/models/")

            pydiffvg.save_svg(os.path.join(cfg.bezier_dir, f'bezier_step_{step}.svg'),
                              painter.canvas_width, painter.canvas_height, painter.bezier_shapes, painter.bezier_shape_groups)

    if cfg.report_to_wandb:
        wandb.finish()
    
    # Saves a high quality .gif from the final SVG frames
    util.save_hq_video(cfg.output_folder, iter_=cfg.num_iter)
