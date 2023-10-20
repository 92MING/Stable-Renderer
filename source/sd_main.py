from sd.modules.data_classes import CorrespondenceMap, ImageFrames
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
from diffusers import EulerAncestralDiscreteScheduler
from sys import platform
import torch
import os

def save_images_as_gif(images: list, output_fname: str = 'output.gif'):
    images[0].save(output_fname, format="GIF", save_all=True, append_images=images[1:], loop=0)

class Config:
    # pipeline init configs
    model_path="runwayml/stable-diffusion-v1-5"
    control_net_model_paths=[
        "lllyasviel/sd-controlnet-depth",
    ]
    # pipeline generation configs
    prompt="boat in van gogh style"
    neg_prompt="low quality, bad anatomy"
    width=512
    height=512
    seed=1234
    device='cuda'
    # data preparation configs
    num_frames=8
    frames_dir="../rendered_frames/maps_per_30_512x512"

if __name__ == '__main__':
    config = Config()

    # 1. Load pipeline
    print("[INFO] Loading pipeline...")
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.model_path,  # Stable Diffusion model path
        control_net_model_paths=config.control_net_model_paths,       use_safetensors=True,
        no_half=(platform == 'darwin')  # Disable fp16 on MacOS
    )
    scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.to(config.device)

    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # 2. Prepare data
    corr_map = CorrespondenceMap.from_existing_directory_img(
        os.path.join(config.frames_dir, 'id'),
        enable_strict_checking=False,
        pixel_position_callback=lambda x,y: (x//8, y//8),
        num_frames=config.num_frames)
    images = ImageFrames.from_existing_directory(
        os.path.join(config.frames_dir, 'color'),
        num_frames=config.num_frames).Data
    depth_images = ImageFrames.from_existing_directory(
        os.path.join(config.frames_dir, 'depth'),
        num_frames=config.num_frames
    ).Data
    controlnet_images = [[img] for img in depth_images]

    # 3. Generate frames
    output_frame_list = pipe.__call__(
        prompt=config.prompt,
        negative_prompt=config.neg_prompt,
        images=images,
        # masks=masks,  # Optional: mask images
        control_images=controlnet_images,
        width=config.width,
        height=config.height,
        num_inference_steps=32,
        strength=1,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.5,
        add_predicted_noise=True,
        correspondence_map=corr_map,
        # callback=utils.view_latents,
    ).images

    # 4. Output 
    output_flattened = [img_list[0] for img_list in output_frame_list]
    save_images_as_gif(images=output_flattened)