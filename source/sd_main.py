from sd.modules.data_classes import CorrespondenceMap, ImageFrames
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
import sd.modules.log_utils as logu
from diffusers import EulerAncestralDiscreteScheduler
from sys import platform
from typing import List
from PIL import Image
import torch
import os
from utils.global_utils import GetEnv
from utils.path_utils import GIF_OUTPUT_DIR
from datetime import datetime


def save_images_as_gif(images: list, output_fname: str = 'output.gif'):
    if not os.path.exists(GIF_OUTPUT_DIR):
        os.makedirs(GIF_OUTPUT_DIR)
    path = os.path.join(GIF_OUTPUT_DIR, datetime.now().strftime(f"%Y-%m-%d_%H-%M_{output_fname}"))
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:], loop=0)
    logu.success(f'[SUCESS] Saved image sequence as {path}')

class Config:
    # pipeline init configs
    model_path="runwayml/stable-diffusion-v1-5"
    control_net_model_paths=[
        "lllyasviel/sd-controlnet-depth",
        "lllyasviel/sd-controlnet-normal",
    ]
    device='mps' if platform == 'darwin' else 'cuda'
    # pipeline generation configs
    prompt="boat in van gogh style"
    neg_prompt="low quality, bad anatomy"
    width=512
    height=512
    seed=1234
    strength=1
    # data preparation configs
    num_frames=8
    frames_dir="../rendered_frames/2023-10-21_13"

if __name__ == '__main__':
    config = Config()

    # 1. Load pipeline
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.model_path,  # Stable Diffusion model path
        control_net_model_paths=config.control_net_model_paths,
        use_safetensors=True,
        torch_dtype=torch.float16,
        device=config.device,
        no_half=(platform == 'darwin')  # Disable fp16 on MacOS
    )
    scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.to(config.device)

    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # 2. Prepare data
    corr_map = CorrespondenceMap.from_existing_directory_numpy(
        os.path.join(config.frames_dir, 'id'),
        enable_strict_checking=False,
        num_frames=config.num_frames,
        use_cache=True)
    images = ImageFrames.from_existing_directory(
        os.path.join(config.frames_dir, 'color'),
        num_frames=config.num_frames).Data
    depth_images = ImageFrames.from_existing_directory(
        os.path.join(config.frames_dir, 'depth'),
        num_frames=config.num_frames
    ).Data
    normal_images = ImageFrames.from_existing_directory(
        os.path.join(config.frames_dir, 'normal'),
        num_frames=config.num_frames
    ).Data
    controlnet_images = [[depth, normal] for depth, normal in zip(depth_images, normal_images)]
    # 3. Generate frames
    output_frame_list = pipe.__call__(
        prompt=config.prompt,
        negative_prompt=config.neg_prompt,
        num_frames=config.num_frames,
        images=images,
        # masks=masks,  # Optional: mask images
        control_images=controlnet_images,
        width=config.width,
        height=config.height,
        num_inference_steps=10,
        strength=config.strength,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.95,
        add_predicted_noise=True, 
        correspondence_map=corr_map,
        overlap_algorithm='resize_overlap',
        callback_kwargs={'save_dir': "./sample"},
        overlap_kwargs={'start_corr': 0, 'end_corr': 600}
        # callback=utils.view_latents,
    ).images

    # 4. Output 
    output_flattened = [img_list[0] for img_list in output_frame_list]
    save_images_as_gif(images=output_flattened)