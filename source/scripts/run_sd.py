import sys, os
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

import torch
import numpy as np
import sd.modules.log_utils as logu

from sys import platform
from dataclasses import dataclass, field
from sd.modules.data_classes import CorrespondenceMap, ImageFrames, Rectangle
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
from sd.modules.diffuser_pipelines.overlap import Overlap, ResizeOverlap, Scheduler
from sd.modules.diffuser_pipelines.overlap.algorithms import overlap_algorithm_factory
from sd.modules.diffuser_pipelines.overlap.scheduler import StartEndScheduler
from sd.modules.diffuser_pipelines.overlap.utils import build_view_normal_map
from sd.modules.diffuser_pipelines.two_step_schedulers import EulerAncestralDiscreteScheduler
from PIL import Image
from typing import List
from datetime import datetime

from utils.global_utils import GetEnv
from utils.path_utils import GIF_OUTPUT_DIR, MAP_OUTPUT_DIR


def save_images_as_gif(images: list, output_fname: str = 'output.gif'):
    if not os.path.exists(GIF_OUTPUT_DIR):
        os.makedirs(GIF_OUTPUT_DIR)
    path = os.path.join(GIF_OUTPUT_DIR, datetime.now().strftime(f"%Y-%m-%d_%H-%M_{output_fname}"))
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:], loop=0)
    logu.success(f'[SUCCESS] Saved image sequence at {path}')

@dataclass
class Config:
    '''pipeline init configs'''
    
    model_path: str = GetEnv('SD_PATH', 'runwayml/stable-diffusion-v1-5')    # type: ignore
    
    control_net_model_paths: List[str] = field(default_factory=list)
    
    device: str = GetEnv('DEVICE', ('mps' if platform == 'darwin' else 'cuda')) # type: ignore
    
    # pipeline generation configs
    prompt: str = GetEnv('DEFAULT_SD_PROMPT', "Golden boat on a calm lake") # type: ignore
    neg_prompt: str = GetEnv('DEFAULT_SD_NEG_PROMPT', "low quality, bad anatomy")   # type: ignore
    width: int = GetEnv('DEFAULT_IMG_WIDTH', 512, int)  # type: ignore
    height: int = GetEnv('DEFAULT_IMG_HEIGHT', 512, int)    # type: ignore
    seed: int = GetEnv('DEFAULT_SEED', 1235, int)   # type: ignore
    no_half: bool = GetEnv('DEFAULT_NO_HALF', False, bool)  # type: ignore
    strength = 1.0
    
    # data preparation configs
    num_frames: int = GetEnv('DEFAULT_NUM_FRAMES',16, int)  # type: ignore
    frames_dir: str = GetEnv('DEFAULT_FRAME_INPUT', "../resources/example-map-outputs/boat")    # type: ignore
    overlap_algorithm: str = 'average'
    start_timestep: int = 0
    end_timestep: int = 1000
    max_workers: int = 1


if __name__ == '__main__':
    control_net_model_paths = [
        # GetEnv('CONTROLNET_LOOSE_DEPTH_MODEL','/research/d1/spc/ckwong1/document/Stable-Renderer/source/sd/models/loose_controlnet'),
        GetEnv('CONTROLNET_DEPTH_MODEL','lllyasviel/sd-controlnet-depth'),
        GetEnv('CONTROLNET_NORMAL_MODEL','lllyasviel/sd-controlnet-normal'),
        # GetEnv('CONTROLNET_CANNY_MODEL','lllyasviel/sd-controlnet-canny'),
    ]   # type: ignore
    
    latest_output = os.path.join(MAP_OUTPUT_DIR, os.listdir(MAP_OUTPUT_DIR)[-1])
    config = Config(control_net_model_paths=control_net_model_paths,    # type: ignore
                    frames_dir = latest_output)

    # 1. Load pipeline
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.model_path,  # Stable Diffusion model path
        control_net_model_paths=config.control_net_model_paths,
        use_safetensors=True,
        torch_dtype=torch.float16,
        device=config.device,
        no_half=config.no_half  # Disable fp16 on MacOS
    )
    scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.to(config.device)

    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # 2. Define overlap algorithm
    alpha_scheduler = Scheduler(
        start_timestep=config.start_timestep, end_timestep=config.end_timestep,
        interpolate_begin=1, interpolate_end=0, power=1, interpolate_type='linear', no_interpolate_return=0)
    corr_map_decay_scheduler = Scheduler(
        start_timestep=750, end_timestep=1000,
        interpolate_begin=1, interpolate_end=1, power=1, interpolate_type='linear', no_interpolate_return=1)
    kernel_radius_scheduler = Scheduler(
        start_timestep=0, end_timestep=1000,
        interpolate_begin=3, interpolate_end=2, power=1, interpolate_type='linear', no_interpolate_return=0)
    scheduled_overlap_algorithm = ResizeOverlap(alpha_scheduler=alpha_scheduler, 
                                                corr_map_decay_scheduler=corr_map_decay_scheduler,
                                                kernel_radius_scheduler=kernel_radius_scheduler,
                                                algorithm=overlap_algorithm_factory(config.overlap_algorithm),  # type: ignore
                                                max_workers=config.max_workers, 
                                                interpolate_mode='nearest')
    
    # 3. Prepare data
    corr_map = CorrespondenceMap.from_existing(
        os.path.join(config.frames_dir, 'id'),
        enable_strict_checking=False,
        num_frames=config.num_frames)
    corr_map.merge_nearby(15)
    # corr_map.dropout_index(probability=0.3, seed=config.seed)
    # corr_map.dropout_in_rectangle(Rectangle((170, 168), (351, 297)), at_frame=0)

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
    try:
        canny_images = ImageFrames.from_existing_directory(
            os.path.join(config.frames_dir, 'canny'),
            num_frames=config.num_frames
        ).Data
    except AssertionError:
        canny_images = None
    if canny_images:
        controlnet_images = [[depth, canny] for depth, normal, canny in zip(depth_images, normal_images, canny_images)]
    else:
        controlnet_images = [[depth, normal] for depth, normal in zip(depth_images, normal_images)]
        # controlnet_images = [*depth_images]
        
    # view_normal_map = build_view_normal_map(normal_images, torch.tensor([0,0,1]))

    # 4. Generate frames
    output_frame_list = pipe.__call__(
        prompt=config.prompt,
        negative_prompt=config.neg_prompt,
        images=images,
        # masks=masks,  # Optional: mask images
        control_images=controlnet_images,
        width=config.width,
        height=config.height,
        num_inference_steps=10,
        strength=config.strength,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.8,
        add_predicted_noise=False,
        correspondence_map=corr_map,
        overlap_algorithm=scheduled_overlap_algorithm,
        callback_kwargs={'save_dir': "./sample"},
        same_init_latents=True,
        same_init_noise=True,
        # view_normal_map=view_normal_map,
        # callback=utils.view_latents,
    ).images

    # 4. Output 
    output_flattened = [img_list[0] for img_list in output_frame_list]
    save_images_as_gif(images=output_flattened, output_fname='output.gif')

    masked_images = []
    for img, depth in zip(output_flattened, depth_images):
        depth = np.array(depth)
        mask = (depth > 0).astype('uint8')
        img_array = np.array(img)
        img_array = img_array * mask[..., None]
        masked_images.append(Image.fromarray(img_array))
        
    save_images_as_gif(images=masked_images, output_fname='masked.gif')