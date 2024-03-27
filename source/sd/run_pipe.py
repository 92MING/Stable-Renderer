import os
import torch
import numpy as np

from sys import platform
from dataclasses import dataclass, field
from sd.modules.data_classes import CorrespondenceMap, ImageFrames, Rectangle
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableRendererPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
from sd.modules.diffuser_pipelines.overlap import Overlap, ResizeOverlap, Scheduler
from sd.modules.diffuser_pipelines.overlap.algorithms import overlap_algorithm_factory
from sd.modules.diffuser_pipelines.overlap.utils import build_view_normal_map
from sd.modules.diffuser_pipelines.schedulers import EulerDiscreteScheduler, LCMScheduler
from PIL import Image
from typing import List, Optional
from datetime import datetime

from common_utils.image_utils import save_images_as_gif
from common_utils.global_utils import GetEnv
from common_utils.debug_utils import DefaultLogger
from common_utils.path_utils import OUTPUT_DIR as GIF_OUTPUT_DIR, MAP_OUTPUT_DIR, EXAMPLE_MAP_OUTPUT_DIR


@dataclass
class RunPipeConfig:
    '''Configs for running the Stable Diffusion model with overlapping'''

    # region pipeline configs
    device: str = GetEnv('DEVICE', ('mps' if platform == 'darwin' else 'cuda')) # type: ignore
    '''GPU device to use. Default is `mps` on MacOS and `cuda` on other platforms.'''
    model_path: str = GetEnv('SD_PATH', 'runwayml/stable-diffusion-v1-5')    # type: ignore
    '''Path(real path or HG's name) to the Stable Diffusion model. Default is `runwayml/stable-diffusion-v1-5`.'''
    prompt: str = GetEnv('DEFAULT_SD_PROMPT', "Golden boat on a calm lake") # type: ignore
    '''Prompt for the diffusion model. This is the main prompt for the model.'''
    neg_prompt: str = GetEnv('DEFAULT_SD_NEG_PROMPT', "low quality, bad anatomy")   # type: ignore 
    '''Negative prompt for the diffusion model. This is the negative prompt for the model.'''
    width: int = GetEnv('DEFAULT_IMG_WIDTH', None)  # type: ignore
    '''Width of the generated image. Default is 512. If None, will use the width of the first frame.'''
    height: int = GetEnv('DEFAULT_IMG_HEIGHT', None)    # type: ignore
    '''Height of the generated image. Default is 512. If None, will use the height of the first frame.'''
    seed: int = GetEnv('DEFAULT_SEED', 1235, int)   # type: ignore
    '''Seed for the random number generator. Default is 1235.'''
    guidance_scale: float = GetEnv('DEFAULT_GUIDANCE_SCALE', 7.5, float)  # type: ignore
    '''Text prompt guidance scale. Higher value means more influence of the prompt.'''
    num_frames: int = GetEnv('DEFAULT_NUM_FRAMES',16, int)  # type: ignore
    '''Number of accepting frames. Default is 16. If set to -1, it will generate unlimited frames.'''
    no_half: bool = GetEnv('DEFAULT_NO_HALF', False, bool)  # type: ignore
    '''Disable fp16 on MacOS. Default is False.'''
    num_inference_steps: int = GetEnv('DEFAULT_NUM_INFERENCE_STEPS', 4, int)  # type: ignore
    '''Number of inference steps to generate the video. Note that this field will be ignored if `specific_timesteps` is not empty'''
    specific_timesteps: Optional[List[int]] = None
    '''
    Specific timesteps to generate. Note that `num_inference_steps` will be ignored if this is not empty.
    When using this, make sure the provided timesteps are accessible in the model, e.g. LCM-Lora could only generate with trained timesteps.
    Supported timesteps for https://civitai.com/models/195519/lcm-lora-weights-stable-diffusion-acceleration-module:
        [ 19  39  59  79  99 119 139 159 179 199 219 239 259 279 299 319 339 359
        379 399 419 439 459 479 499 519 539 559 579 599 619 639 659 679 699 719
        739 759 779 799 819 839 859 879 899 919 939 959 979 999]
    '''
    enable_ancestral_sampling: bool = GetEnv('ENABLE_ANCESTRAL_SAMPLING', False, bool)  # type: ignore
    '''Enable ancestral sampling. Ancestral sampling means adding new random noise to each latent after each timestep. Default is False.'''
    disable_ancestral_at_last: bool = GetEnv('DISABLE_ANCESTRAL_AT_LAST', False, bool)  # type: ignore
    '''Disable ancestral sampling at the last timestep. This is only for `enable_ancestral_sampling=True`. Default is False.'''
    same_init_latents: bool = True
    '''Whether to use the same initial random latents for all frames. 
    This setting is only for `do_img2img`=False (means that you do not give base reference images(known as engine's color output)).'''
    same_init_noise: bool = True
    '''Whether to use the same initial random noise for all frames.
    This setting is only for `do_img2img`=True (means that you give base reference images(known as engine's color output)).'''
    # endregion

    # region img2img configs
    do_img2img: bool = True
    '''enable img2img'''
    img2img_strength: float = 0.8
    '''strength of img referencing'''
    # endregion
    
    # region inpainting configs
    do_inpainting: bool = True
    '''enable inpainting'''
    # endregion
    
    # region controlnet configs
    do_controlnet: bool = True
    '''enable controlnet'''
    controlnet_depth_model: Optional[str] = GetEnv('CONTROLNET_DEPTH_MODEL','lllyasviel/sd-controlnet-depth')
    '''depth controlnet model path. If None, depth control will be disabled.'''
    controlnet_normal_model: Optional[str] = GetEnv('CONTROLNET_NORMAL_MODEL','lllyasviel/sd-controlnet-normal')
    '''normal controlnet model path. If None, normal control will be disabled.'''
    controlnet_canny_model: Optional[str] = GetEnv('CONTROLNET_CANNY_MODEL','lllyasviel/sd-controlnet-canny')
    '''canny controlnet model path. If None, canny control will be disabled.'''
    controlnet_loose_depth_model: Optional[str] = GetEnv('CONTROLNET_LOOSE_DEPTH_MODEL', None)
    '''Loose Control. This path should actually be a directory containing two files: loose_controlnet.safetensors and config.json.
    You can download them from: https://huggingface.co/AIRDGempoll/LooseControlNet/tree/main'''
    other_controlnet_model_paths: List[str] = field(default_factory=list)
    '''extra controlnet model paths. If None, extra control will be disabled.'''
    # endregion
    
    # region lora configs
    use_lcm_lora: bool = True
    '''enable LCM-Lora'''
    lcm_lora_path: str = GetEnv('LCM_LORA_PATH', 'latent-consistency/lcm-lora-sdv1-5')  # type: ignore
    '''Path to the LCM-Lora model. Default is `latent-consistency/lcm-lora-sdv1-5`.'''
    # endregion
    
    # region overlap configs
    do_overlapping: bool = True
    '''allow overlapping of frames'''
    ignore_first_overlap: bool = GetEnv('IGNORE_FIRST_OVERLAP', False, bool)  # type: ignore
    '''If skip the first overlap, e.g. skipping timestep=999 (if first timestep is 999)'''
    kernel_radius_start_value: int = GetEnv('KERNEL_RADIUS', 2, int)  # type: ignore
    '''Radius of the kernel for the overlap algorithm. Default is 1.'''
    kernel_radius_end_value: int = GetEnv('KERNEL_RADIUS', 0, int)  # type: ignore
    '''Radius of the kernel for the overlap algorithm. Default is 0.'''
    overlap_algorithm: str = 'average'
    '''How should latent values be mixed. Default is `average`.'''
    corrmap_merge_len: int = 4
    '''Resize the correspondence map to become smaller by merging nearby points. It helps increasing the point matching rate. 
    Default is 4. If set to 0, it will not merge any points.'''
    overlap_start_timestep: int = 0
    overlap_end_timestep: int = 1000
    overlap_start_value: float = 1.0
    overlap_end_value: float = 0.0
    max_workers: int = 1    # TODO: multi-worker overlap
    # endregion
    
    # region IO
    frames_dir: Optional[str] = GetEnv('DEFAULT_FRAME_INPUT', os.path.join(EXAMPLE_MAP_OUTPUT_DIR, "boat"))    # type: ignore
    '''Path to the directory containing the frames. Default is `../resources/example-map-outputs/boat`.
    If set to None, will try to get the latest output from `MAP_OUTPUT_DIR`.'''
    save_gif: bool = GetEnv('SAVE_GIF', True, bool)  # type: ignore
    '''Save the generated gif. Default is True.'''
    gif_output_dir: str = GetEnv('GIF_OUTPUT_DIR', GIF_OUTPUT_DIR)  # type: ignore
    '''Path to the directory to save the gif. Default is `output/gifs`.'''
    save_latent: bool = GetEnv('SAVE_LATENT', False, bool)  # type: ignore
    '''Save the latent for each timestep (for debugging). Default is False.'''
    # endregion



def run_pipe(config: RunPipeConfig = None):
    if not config:
        config = RunPipeConfig()
    
    # 1. Prepare data
    if config.frames_dir is None:   # get the latest output from MAP_OUTPUT_DIR
        config.frames_dir = os.path.join(MAP_OUTPUT_DIR, os.listdir(MAP_OUTPUT_DIR)[-1])
    
    corr_map = CorrespondenceMap.from_existing(
        os.path.join(config.frames_dir, 'id'),
        enable_strict_checking=False,
        num_frames=config.num_frames)
    
    if config.corrmap_merge_len:
        corr_map.merge_nearby(config.corrmap_merge_len)
    # corr_map.dropout_index(probability=0.3, seed=config.seed)
    # corr_map.dropout_in_rectangle(Rectangle((170, 168), (351, 297)), at_frame=0)

    if config.num_frames == 1:
        DefaultLogger.warning("Number of frames is 1. Overlapping will be disabled.")
        config.do_overlapping = False
    
    if config.do_img2img:
        color_images = ImageFrames.from_existing_directory(
            directory=os.path.join(config.frames_dir, 'color'),
            num_frames=config.num_frames).Data
    else:
        color_images = None
    
    imgs_for_gif_masking = None  # for building masked gif.
    
    if config.do_controlnet:
        
        if config.controlnet_depth_model or config.controlnet_loose_depth_model:
            try:
                depth_images = ImageFrames.from_existing_directory(
                    directory=os.path.join(config.frames_dir, 'depth'),
                    num_frames=config.num_frames
                ).Data
                if not imgs_for_gif_masking:
                    imgs_for_gif_masking = depth_images
            except AssertionError:
                DefaultLogger.warning("Normal images are not found. Depth control will be disabled.")
                depth_images = None
                config.controlnet_depth_model = None
                config.controlnet_loose_depth_model = None
        else:
            depth_images = None
        
        if config.controlnet_normal_model:
            try:
                normal_images = ImageFrames.from_existing_directory(
                    directory=os.path.join(config.frames_dir, 'normal'),
                    num_frames=config.num_frames
                ).Data
                if not imgs_for_gif_masking:
                    imgs_for_gif_masking = normal_images
            except AssertionError:
                DefaultLogger.warning("Normal images are not found. Normal control will be disabled.")
                normal_images = None
                config.controlnet_normal_model = None
        else:
            normal_images = None
        
        if config.controlnet_canny_model:
            try:
                canny_images = ImageFrames.from_existing_directory(
                    os.path.join(config.frames_dir, 'canny'),
                    num_frames=config.num_frames
                ).Data
            except AssertionError:
                DefaultLogger.warning("Canny images are not found. Canny control will be disabled.")
                canny_images = None
                config.controlnet_canny_model = None
        else:
            canny_images = None
        
        controlnet_images = [imgs for imgs in [depth_images, normal_images, canny_images] if imgs is not None]
        
        min_frame_len = min([len(img) for img in controlnet_images])
        if min_frame_len > config.num_frames:
            DefaultLogger.info(f"Number of frames given({min_frame_len}) in controlnet images is greater than {config.num_frames}. Truncating to {config.num_frames}.")
        elif min_frame_len < config.num_frames:
            DefaultLogger.info(f"Number of frames given({min_frame_len}) in controlnet images is less than {config.num_frames}. Padding to {min_frame_len}.")
            config.num_frames = min_frame_len
        
        controlnet_images = [img[:config.num_frames] for img in controlnet_images]
            
        if len(controlnet_images) == 1:
            controlnet_images = [*controlnet_images[0]]
        else:
            new_controlnet_images = []
            for i in range(config.num_frames):
                img_list = [imgs[i] for imgs in controlnet_images]
                new_controlnet_images.append(img_list)
            controlnet_images = new_controlnet_images

    else:
        controlnet_images = None
        if color_images:
            imgs_for_gif_masking = color_images
    
    if config.controlnet_depth_model and config.controlnet_loose_depth_model:
        raise ValueError("Currently you cannot use both controlnet_depth_model and controlnet_loose_depth_model at the same time. Specify only one.")
    
    controlnet_model_paths = []
    if config.do_controlnet:
        if config.controlnet_depth_model:
            controlnet_model_paths.append(config.controlnet_depth_model)
        
        if config.controlnet_loose_depth_model:
            controlnet_model_paths.append(config.controlnet_loose_depth_model)
            
        if config.controlnet_normal_model:
            controlnet_model_paths.append(config.controlnet_normal_model)
        
        if config.controlnet_canny_model:
            controlnet_model_paths.append(config.controlnet_canny_model)
        
        if config.other_controlnet_model_paths:
            controlnet_model_paths.extend(config.other_controlnet_model_paths)
            
    if not controlnet_model_paths and config.do_controlnet:
        DefaultLogger.warning("Controlnet is enabled but no model is provided. Controlnet will be disabled.")
        config.do_controlnet = False
    
    if config.specific_timesteps:
        config.specific_timesteps.sort(reverse=True)    # timestep should be in descending order

    # 2. Load pipeline
    if config.specific_timesteps:
        if config.use_lcm_lora is False:
            raise ValueError(
                "specific_timesteps is provided but use_lcm_lora is False. Please enable use_lcm_lora in config."
            )
        if config.num_inference_steps > 0:
            DefaultLogger.warning(
                "Both specific_timesteps and num_inference_steps are provided. num_inference_steps will be ignored."
            )
        config.num_inference_steps = len(config.specific_timesteps)
    
    pipe: StableRendererPipeline = load_pipe(
        model_path=config.model_path,  # Stable Diffusion model path
        control_net_model_paths=controlnet_model_paths,  # Controlnet model paths
        use_safetensors=True,
        torch_dtype=torch.float16,
        device=config.device,
        no_half=config.no_half  # Disable fp16 on MacOS
    )
    
    if config.use_lcm_lora and not config.lcm_lora_path:
        DefaultLogger.warning("LCM-Lora is enabled but no path is provided. LCMLora will be disabled.")
        config.use_lcm_lora = False
    
    if not config.use_lcm_lora:
        scheduler: EulerDiscreteScheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, 
                                                                               enable_ancestral_sampling = config.enable_ancestral_sampling,
                                                                                 disable_ancestral_at_last=config.disable_ancestral_at_last,
                                                                                 img2img=config.do_img2img)    # type: ignore
    else:
        pipe.load_lora_weights(config.lcm_lora_path)
        scheduler: LCMScheduler = LCMScheduler.from_config(pipe.scheduler.config, 
                                                        enable_ancestral_sampling = config.enable_ancestral_sampling,
                                                        disable_ancestral_at_last=config.disable_ancestral_at_last,
                                                        img2img=config.do_img2img)  # type: ignore
    
    pipe.scheduler = scheduler
    pipe.to(config.device)

    generator = torch.Generator(device=config.device).manual_seed(config.seed)


    # 3. Define overlap algorithm
    alpha_scheduler = Scheduler(start_timestep=config.overlap_start_timestep, 
                                end_timestep=config.overlap_end_timestep,
                                interpolate_begin=config.overlap_start_value, 
                                interpolate_end=config.overlap_end_value, 
                                power=1, 
                                interpolate_type='linear', 
                                no_interpolate_return=0)
    corr_map_decay_scheduler = Scheduler(start_timestep=750, 
                                         end_timestep=1000,
                                         interpolate_begin=1, 
                                         interpolate_end=1, 
                                         power=1, 
                                         interpolate_type='linear', 
                                         no_interpolate_return=1)
    kernel_radius_scheduler = Scheduler(start_timestep=0, 
                                        end_timestep=1000,
                                        interpolate_begin=config.kernel_radius_start_value,
                                        interpolate_end=config.kernel_radius_end_value,
                                        power=1, 
                                         interpolate_type='linear', 
                                        no_interpolate_return=0)
    scheduled_overlap_algorithm = ResizeOverlap(alpha_scheduler=alpha_scheduler, 
                                                corr_map_decay_scheduler=corr_map_decay_scheduler,
                                                kernel_radius_scheduler=kernel_radius_scheduler,
                                                algorithm=overlap_algorithm_factory(config.overlap_algorithm),  # type: ignore
                                                max_workers=config.max_workers, 
                                                interpolate_mode='nearest')
    
    
        
    # view_normal_map = build_view_normal_map(normal_images, torch.tensor([0,0,1]))

    # 4. Generate frames
    start_time = datetime.now()
    
    if not config.width:
        if imgs_for_gif_masking:
            config.width = imgs_for_gif_masking[0].width
        else:
            DefaultLogger.warning("Width is not provided and no images are given for gif masking. Defaulting to 512.")
            config.width = 512
    if not config.height:
        if imgs_for_gif_masking:
            config.height = imgs_for_gif_masking[0].height
        else:
            DefaultLogger.warning("Height is not provided and no images are given for gif masking. Defaulting to 512.")
            config.height = 512
    
    if isinstance(config.width, str):
        config.width = int(config.width)
    if isinstance(config.height, str):
        config.height = int(config.height)
    
    output_frame_list = pipe.__call__(
        prompt=config.prompt,
        negative_prompt=config.neg_prompt,
        images=color_images,
        # masks=masks,  # Optional: mask images
        control_images=controlnet_images,
        width=config.width,
        height=config.height,
        num_frames=config.num_frames,
        
        num_inference_steps=config.num_inference_steps,
        specific_timesteps=config.specific_timesteps,
        img2img_strength=config.img2img_strength,
        generator=generator,
        
        guidance_scale=config.guidance_scale,
        controlnet_conditioning_scale=0.8,
        add_predicted_noise=False,
        correspondence_map=corr_map,
        overlap_algorithm=scheduled_overlap_algorithm,
        
        # callback_kwargs={'save_dir': "./sample"},
        same_init_latents=config.same_init_latents,
        same_init_noise=config.same_init_noise,
        save_latent=config.save_latent,
        ignore_first_overlap=config.ignore_first_overlap,
        # view_normal_map=view_normal_map,
        # callback=utils.view_latents,
        
        do_img2img=config.do_img2img,
        do_inpainting=config.do_inpainting,
        do_controlnet=config.do_controlnet,
        do_overlapping=config.do_overlapping,
    ).images
    
    time_usage = datetime.now() - start_time

    # 4. Output 
    if config.save_gif:
        output_flattened = [img_list[0] for img_list in output_frame_list]
        save_images_as_gif(images=output_flattened, output_fname='output.gif')

        masked_images = []
        if imgs_for_gif_masking:
            for img, depth in zip(output_flattened, imgs_for_gif_masking):
                depth = np.array(depth)
                mask = (depth > 0).astype('uint8')
                img_array = np.array(img)
                img_array = img_array * mask[..., None]
                masked_images.append(Image.fromarray(img_array))
            save_images_as_gif(images=masked_images, output_fname='masked.gif')
        else:
            DefaultLogger.info("No images for gif masking. Masked gif will not be saved.")
    
    
    
    print(f"Time usage: {time_usage.total_seconds()}s")
    
    
    
__all__ = ['RunPipeConfig', 'run_pipe']
