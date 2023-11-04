import sys, os
sys.path.append(os.getcwd())

from sd.modules.data_classes import CorrespondenceMap, ImageFrames
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
from sd.modules.diffuser_pipelines.overlap import StartEndScheduler, AlphaScheduler, Overlap, ResizeOverlap, VAEOverlap
import sd.modules.log_utils as logu
from diffusers import EulerAncestralDiscreteScheduler
from sys import platform
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
    logu.success(f'[SUCESS] Saved image sequence at {path}')

class Config:
    # pipeline init configs
    model_path=GetEnv('SD_PATH', 'runwayml/stable-diffusion-v1-5')
    control_net_model_paths=[
        GetEnv('CONTROLNET_DEPTH_MODEL','lllyasviel/sd-controlnet-depth'),
        GetEnv('CONTROLNET_NORMAL_MODEL','lllyasviel/sd-controlnet-normal'),
    ]
    device = GetEnv('DEVICE', ('mps' if platform == 'darwin' else 'cuda'))
    # pipeline generation configs
    prompt = GetEnv('DEFAULT_SD_PROMPT', "boat in van gogh style")
    neg_prompt = GetEnv('DEFAULT_SD_NEG_PROMPT', "low quality, bad anatomy")
    width = GetEnv('DEFAULT_IMG_WIDTH', 512, int)
    height = GetEnv('DEFAULT_IMG_HEIGHT', 512, int)
    seed = GetEnv('DEFAULT_SEED', 1234, int)
    strength = 1
    # data preparation configs
    num_frames = GetEnv('DEFAULT_NUM_FRAMES', 8, int)
    frames_dir = GetEnv('DEFAULT_FRAME_INPUT', "../rendered_frames/2023-10-21_13")
    # Overlap algorithm configs
    alpha = 0.5
    max_workers = 1
    start_corr = 600
    end_corr = 1000

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

    # 2. Define overlap algorithm
    overlap_algorithm = VAEOverlap(
        vae=pipe.vae, generator=generator, alpha=config.alpha, max_workers=config.max_workers)
    scheduled_overlap_algorithm = AlphaScheduler(
        alpha_start=0.8, alpha_end=0.4, schedule_mode='linear', overlap=overlap_algorithm)

    # 3. Prepare data
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

    # 4. Generate frames
    output_frame_list = pipe.__call__(
        prompt=config.prompt,
        negative_prompt=config.neg_prompt,
        num_frames=config.num_frames,
        images=images,
        # masks=masks,  # Optional: mask images
        control_images=controlnet_images,
        width=config.width,
        height=config.height,
        num_inference_steps=5,
        strength=config.strength,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.5,
        add_predicted_noise=True, 
        correspondence_map=corr_map,
        overlap_algorithm=scheduled_overlap_algorithm,
        callback_kwargs={'save_dir': "./sample"},
        same_init_latents=False
        # callback=utils.view_latents,
    ).images

    # 4. Output 
    output_flattened = [img_list[0] for img_list in output_frame_list]
    save_images_as_gif(images=output_flattened)