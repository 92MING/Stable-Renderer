import torch
from typing import Literal, TypeAlias

from comfyUI.types import *
from comfyUI.nodes import custom_ksampler
from common_utils.debug_utils import ComfyUILogger
from stable_rendering.src.data_classes import CorrespondenceMap
from stable_rendering.src.overlap import ResizeOverlap, Scheduler, overlap_algorithm_factory


_default_sampler = COMFY_SAMPLERS.__args__[0]   # type: ignore
_default_scheduler = COMFY_SCHEDULERS.__args__[0]   # type: ignore

OverlapAlgorithm: TypeAlias = Literal["average", "frame_distance", "pixel_distance", "perpendicular_view_normal"]

class StableRenderSampler(StableRenderingNode):
    
    Category = "sampling"

    def __call__(self, 
                 model: MODEL,
                 positive: "CONDITIONING", # type: ignore
                 negative: "CONDITIONING", # type: ignore
                 latent_image: LATENT,
                 correspondence_map: CorrespondenceMap,
                 alpha_scheduler: Scheduler,
                 kernel_radius_scheduler: Scheduler,
                 overlap_algorithm: OverlapAlgorithm = "average",
                 noise_option: Literal['disable', 'default', 'incoming'] = 'default',
                 apply_overlap_option: Literal['noise', 'denoised', 'both'] = 'noise',
                 noise_seed: INT(0, 0xffffffffffffffff)=0, # type: ignore
                 steps: INT(1, 10000)=20, # type: ignore
                 cfg: FLOAT(0.0, 100.0, 0.01, round=0.01)=8.0, # type: ignore
                 sampler_name: COMFY_SAMPLERS=_default_sampler,
                 scheduler: COMFY_SCHEDULERS=_default_scheduler,
                 denoise: FLOAT(0, 1) = 1.0 # type: ignore
                 )->LATENT:
        """
        This method is responsible for sampling with Correspondence Map.

        Args:
            model (MODEL): The model to render.
            positive (CONDITIONING): The positive conditioning.
            negative (CONDITIONING): The negative conditioning.
            latent_image (LATENT): The latent image.
            correspondence_map (CorrespondenceMap): The correspondence map.
            alpha_scheduler (Scheduler): The alpha scheduler.
            kernel_radius_scheduler (Scheduler): The kernel radius scheduler.
            overlap_algorithm (Literal[
                'average', 'frame_distance', 'pixel_distance', 'perpendicular_view_normal'
            ], optional): The overlap algorithm. Defaults to "average".
            noise_option (Literal['disable', 'default', 'incoming'], optional):
                The noise option. Defaults to 'default'. If 'disable', noise is not used. 
                If 'default', noise is used. If 'incoming', noise is used and the incoming noise is used.
            apply_overlap_option (Literal['noise', 'denoised', 'both'], optional): The apply overlap option. Defaults to 'noise'.
            noise_seed (INT(0, 0xffffffffffffffff), optional): The noise seed. Defaults to 0.
            steps (INT(1, 10000), optional): The number of steps. Defaults to 20.
            cfg (FLOAT(0.0, 100.0, 0.01, round=0.01), optional): The cfg value. Defaults to 8.0.
            sampler_name (COMFY_SAMPLERS, optional): The sampler name. Defaults to _default_sampler.
            scheduler (COMFY_SCHEDULERS, optional): The scheduler. Defaults to _default_scheduler.
            denoise (FLOAT(0, 1), optional): The denoise value. Defaults to 1.0.

        Returns:
            LATENT: The sampled latent image.
        """
        SUPPORTED_SAMPLERS = ["ddim", "ddpm"]

        if sampler_name not in SUPPORTED_SAMPLERS:
            ComfyUILogger.warning(f"Scheduling with {sampler_name} is not supported. Using default sampler instead.")
            sampler_name = "ddpm"

        callbacks = []
        overlap = ResizeOverlap(
            alpha_scheduler=alpha_scheduler,
            kernel_radius_scheduler=kernel_radius_scheduler,
            algorithm=overlap_algorithm_factory(overlap_algorithm),
        )

        def execute_overlap_on_noise(context: SamplingCallbackContext):
            # Modifies context.noise in place
            noise = context.noise
            frame_seq = [frame.unsqueeze(0) for frame in noise]
            estimated_denoising_timestep = 1000 - int(((context.step_index + 1) / context.total_steps) * 1000)
            overlapped_frame_seq = overlap.__call__(
                frame_seq,
                corr_map=correspondence_map,
                step=context.step_index,
                timestep=estimated_denoising_timestep
            )
            for i, frame in enumerate(overlapped_frame_seq):
                context.noise[i] = frame.squeeze()

        def execute_overlap_on_denoised(context: SamplingCallbackContext):
            # Modifies context.denoised in place
            denoised = context.denoised
            frame_seq = [frame.unsqueeze(0) for frame in denoised]
            estimated_denoising_timestep = 1000 - int(((context.step_index + 1) / context.total_steps) * 1000)
            overlapped_frame_seq = overlap.__call__(
                frame_seq,
                corr_map=correspondence_map,
                step=context.step_index,
                timestep=estimated_denoising_timestep
            )
            for i, frame in enumerate(overlapped_frame_seq):
                context.denoised[i] = frame.squeeze()

        def execute_overlap(context: SamplingCallbackContext):
            match apply_overlap_option:
                case 'noise':
                    execute_overlap_on_noise(context)
                case 'denoised':
                    if sampler_name != "ddpm":
                        ComfyUILogger.warning(
                            f"apply_overlap_option is set to 'denoised' but sampler_name is not 'ddpm'. Using on noise version instead."
                        )
                        execute_overlap_on_noise(context)
                    else:
                        execute_overlap_on_denoised(context)
                case 'both':
                    if sampler_name != "ddpm":
                        ComfyUILogger.warning(
                            f"apply_overlap_option is set to 'both' but sampler_name is not 'ddpm'. Using on noise version instead."
                        )
                        execute_overlap_on_noise(context)
                    else:
                        execute_overlap_on_noise(context)
                        execute_overlap_on_denoised(context)
                case _:
                    raise ValueError(f"Unknown apply_overlap_option: {apply_overlap_option}")

        callbacks.append(execute_overlap)

        return custom_ksampler(model,
                               noise_seed,
                               steps,
                               cfg,
                               sampler_name,
                               scheduler,
                               positive,
                               negative,
                               latent_image,
                               denoise=denoise,
                               noise_option=noise_option,
                               callbacks=callbacks)
