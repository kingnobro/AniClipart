import torch
import pydiffvg
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import cairosvg

# ==================================
# ====== video realted utils =======
# ==================================
def frames_to_vid(video_frames, output_vid_path):
    """
    Saves an mp4 file from the given frames
    """
    writer = imageio.get_writer(output_vid_path, fps=8)
    for im in video_frames:
        writer.append_data(im)
    writer.close()

def render_frames_to_tensor(frames_shapes, frames_shapes_grous, w, h, render, device):
    """
    Given a list with the points parameters, render them frame by frame and return a tensor of the rasterized frames ([16, 256, 256, 3])
    """
    frames_init = []
    for i in range(len(frames_shapes)):
        shapes = frames_shapes[i]
        shape_groups = frames_shapes_grous[i]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        cur_im = render(w, h, 2, 2, 0, None, *scene_args)
    
        cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
               torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=device) * (1 - cur_im[:, :, 3:4])
        cur_im = cur_im[:, :, :3]
        frames_init.append(cur_im)
    return torch.stack(frames_init)

def save_mp4_from_tensor(frames_tensor, output_vid_path):
    # input is a [16, 256, 256, 3] video
    frames_copy = frames_tensor.clone()
    frames_output = []
    for i in range(frames_copy.shape[0]):
        cur_im = frames_copy[i]
        cur_im = cur_im[:, :, :3].detach().cpu().numpy()
        cur_im = (cur_im * 255).astype(np.uint8)
        frames_output.append(cur_im)
    frames_to_vid(frames_output, output_vid_path=output_vid_path)
    
def save_vid_svg(frames_svg, output_folder, step, w, h):
    if not os.path.exists(f"{output_folder}/svg_step{step}"):
        os.mkdir(f"{output_folder}/svg_step{step}")
    for i in range(len(frames_svg)):
        pydiffvg.save_svg(f"{output_folder}/svg_step{step}/frame{i:03d}.svg", w, h, frames_svg[i][0], frames_svg[i][1])

def svg_to_png(path_to_svg_files, dest_path):
    svgs = sorted(os.listdir(path_to_svg_files))
    filenames = [k for k in svgs if "svg" in k]
    for filename in filenames:        
        dest_path_ = f"{dest_path}/{os.path.splitext(filename)[0]}.png"
        cairosvg.svg2png(url=f"{path_to_svg_files}/{filename}", write_to=dest_path_, scale=4, background_color="white")
  
def save_gif_from_pngs(path_to_png_files, gif_dest_path, fps=8):
    pngs = sorted(os.listdir(path_to_png_files))
    filenames = [k for k in pngs if "png" in k]
    images = []
    for filename in filenames:
        im = imageio.imread(f"{path_to_png_files}/{filename}")
        images.append(im)
    imageio.mimsave(f"{gif_dest_path}", images, 'GIF', loop=0, fps=fps)

def save_hq_video(path_to_outputs, iter_=1000, is_last_iter=False, fps=8):
    dest_path_png = f"{path_to_outputs}/png_logs/png_files_ite{iter_}"
    os.makedirs(dest_path_png, exist_ok=True)

    svg_to_png(f"{path_to_outputs}/svg_logs/svg_step{iter_}", dest_path_png)

    gif_dest_path = f"{path_to_outputs}/mp4_logs/HQ_gif_iter{iter_}.gif"
    save_gif_from_pngs(dest_path_png, gif_dest_path, fps)
    print(f"GIF saved to [{gif_dest_path}]")

    if is_last_iter:
        gif_dest_path = f"{path_to_outputs}/HQ_gif.gif"
        save_gif_from_pngs(dest_path_png, gif_dest_path, fps)

def normalize_tensor(tensor: torch.Tensor, canvas_size: int = 256):
    range_value = float(canvas_size)# / 2
    normalized_tensor = tensor / range_value
    return normalized_tensor

def get_clipart_caption(target):
    files_to_captions = {
        "man": "A man walks forward.",
        "man_jump": "A young man jumps up and down.",
        "man_wave": "A young man is waving his arms to say hello.",
        "man_run": "The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward.",
        "woman_dance": "A woman dancer is dancing the Cha-Cha.",
        "woman_dance2": "A woman dancer is dancing with her legs moving up and down, and waving her hands",
        "woman_jump": "A young girl jumps up and down.",
        "woman_wave": "A young girl is waving her arms to say goodbye.",
        "woman_dance3": "A woman in a green dress with black polka dots and black boots is dancing joyfully.",
        "woman_dance4": "A woman in a flowing dance move, squatting slightly with her legs, wearing a red top, blue skirt, and black ankle boots.",
        "man_dance": "A bearded man is dancing with the music, his legs squatting up and down slightly, his hands waving in the air.",
        "man_dance2": "A person wearing a pumpkin mask, pink tutu, and orange socks is in an energetic dance with arms raised.",
        "man_sport": "An elderly man is lifting dumbbells up and down in a rhythmic motion.",
        "woman_sport": "An elderly woman with white hair is squatting up and down dramatically. Her hands are in a horizontal position to keep balance.",
        "ninja_kick": "A ninja is performing a high kick, with one leg moved up and the other leg bent. His hands are in a fighting position.",
        "man_fencing": "A fencer in en garde position, ready to advance.",
        "woman_book": "A woman standing while holding a book in her hand.",
        "woman_selfie": "A woman is taking a selfie with her phone while waving the hand that holds an ice cream cone. Her feet are stationary.",
        "woman_yoga": "A woman is practicing yoga, bending her torso back and forth while extending her legs vertically.",
        "seal": "A seal is floating up and down in the water, waving its flippers and tail",
        "turtle": "A turtle floats up and down in the water, extending and retracting its legs.",
        "dolphin": "A dolphin swimming and leaping out of the water, bending its body flexibly.",
        "crab": "A crab is waving its pincers and legs continuously.",
        "shrimp": "A shrimp is swmming and swaying its tentacles.",
        "flower": "A flower sways its petals in the breeze.",
        "flower2": "A flower sways its petals and leaves.",
        "cloud": "A cloud floats in the sky.",
        "chicken": "A chicken is jumping up and down.",
        "bug": "a cheerful green caterpillar is arching and contracting its body.",
        "bird": "A bird is flying up and down.",
        "man_boxing": "Boxing guy is punching and dodging.",
        "starfish": "A starfish is waving its tentacles softly.",
        "elephant": "An elephant jumps and wags its trunk up and down continuously.",
        "kite": "A kite is floating in the sky.",
        "balloon": "A balloon floating in the air.",
        "man_table": "A man playing table tennis is swinging the racket.",
        "man_dive": "A man is scuba diving and swaying fins.",
        "man_dive2": "A diver in mid-air, bending his body and legs to dive into the water.",
        "woman_surf": "A surfer riding and maneuvering on waves on a surfboard.",
        "woman_ballet": "The ballerina is dancing, her arms bending and stretching up and down gracefully.",
        "butterfly": "A butterfly fluttering its wings and flying gracefully.",
        "lemur": "A lemur is bending its long tail continuously.",
        "leopard": "A black leopard is standing and shaking its tail.",
        "stork": "A stork bends its long neck.",
        "candle": "A candle with flickering flame.",
        "bat": "A bat is flapping its wings up and down.",
        "jellyfish": "A jellyfish is floating up and down in the water, speading and swaying its tentacles.",
        "octopus": "An octopus is swimming and swaying its tentacles.",
        "snowman": "A snowman is waving its arms cheerfully.",
        "bird2": "A bird is flappping its wings to make balance in the air.",
        "cat": "A black is playing and waving its long tail from left to right.",
        "bat2": "A bat is flapping its wings up and down.",
        "fish": "The fish is gracefully moving through the water, its fins and tail fin gently propelling it forward with effortless agility.",
        "dog": "A dog is running.",
        "spider": "A spider sways its legs.",
        "flamingo": "A flamingo is walking.",
        "raccoon": "The raccoon is playing.",
        "worm": "The worm is arching and contracting its body.",
        "man_demon": "The man in demon costume is cheering, waving both arms up and down.",
        "woman_speak": "The girl speaks into a megaphone",
        "woman_exercise": "A young girl is exercising in lunging position, lifting dumbbells.",
        "balloon_dog": "A galloping dog",
        "spaceship": "The spaceship accelerates rapidly during takeoff, flies into the sky.",
        "ghost": "The ghost is dancing",
        "flame": "A burning flame sways from side to side.",
        "white_cloud": "The cloud floats in the sky.",
        "flag": "A waving flag fluttering and rippling in the wind.",
        "palm": "The palm tree sways the leaves in the wind.",
        "dragonfly": "A dragonfly is flappping its wings to make balance in the air.",
        "snail": "A snail is moving.",
        "dolphin2": "A dolphin swimming and leaping out of the water, bending its body flexibly.",
        "parrot": "The parrot flapping its wings.",
        "firecracker": "The firecracker flies into the sky.",
        "robot": "A robot is dancing",
        "monster3": "A yello creature is dancing.",
        "parachute": "A parachute descending slowly and gracefully after being deployed.",
        "flower4": "The flower is moving and growing, swaying gently from side to side.",
        "ghost2": "The Halloween ghost is cheering and waving its arms.",
        "ginger": "The gingerbread man is dancing.",
        "woman_phone": "A woman is talking on the phone.",
        "man_boxing2": "The boxer is punching.",
        "man_breaking": "Breakdancer performing an inverted freeze with one hand on the floor and legs artistically intertwined in the air.",
        "man_rap": "A rapper is singing, moving his hands and body to the rhythm of the music.",
        "man_skiing": "A man is skiing down the slope.",
        "bird3": "A bird hovers in mid-air, flapping wings energetically.",
        "camel": "A camel is walking.",
        "man_eat": "A man is eating, moving his hand to his mouth.",
        "woman_swim": "A woman is swimming, bending her arms and legs continuously.",
        "woman_gymnastics": "A woman is doing gymnastics, the ribbon flowing flexibly in the air.",
    }
    return files_to_captions[os.path.basename(target)]