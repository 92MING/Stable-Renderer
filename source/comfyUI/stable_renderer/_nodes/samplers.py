import torch
from comfyUI.types import *
from comfyUI.nodes import custom_ksampler

from common_utils.debug_utils import ComfyUILogger

from stable_renderer.src.data_classes import CorrespondenceMap
from stable_renderer.src.overlap import ResizeOverlap, Scheduler, overlap_algorithm_factory


_default_sampler = COMFY_SAMPLERS.__args__[0]   # type: ignore
_default_scheduler = COMFY_SCHEDULERS.__args__[0]   # type: ignore


class StableRenderSampler(StableRendererNodeBase):
    
    Category = "sampling"

    def __call__(self, 
                 model: MODEL,
                 positive: "CONDITIONING", # type: ignore
                 negative: "CONDITIONING", # type: ignore
                 latent_image: LATENT,
                 correspondence_map: CorrespondenceMap,
                 alpha_scheduler: Scheduler,
                 kernel_radius_scheduler: Scheduler,
                 overlap_algorithm: Literal[
                     "average", "frame_distance", "pixel_distance", "perpendicular_view_normal"
                 ] = "average",
                 noise_option: Literal['disable', 'default', 'incoming'] = 'default',
                 noise_seed: INT(0, 0xffffffffffffffff)=0, # type: ignore
                 steps: INT(1, 10000)=20, # type: ignore
                 cfg: FLOAT(0.0, 100.0, 0.01, round=0.01)=8.0, # type: ignore
                 sampler_name: COMFY_SAMPLERS=_default_sampler,
                 scheduler: COMFY_SCHEDULERS=_default_scheduler,
                 denoise: FLOAT(0, 1) = 1.0)->LATENT:
        SUPPORTED_SAMPLERS = ["ddim", "ddpm"]

        if sampler_name not in SUPPORTED_SAMPLERS:
            ComfyUILogger.warning(f"Scheduling with {sampler_name} is not supported. Using default sampler instead.")
            sampler_name = "ddim"

        callbacks = []
        overlap = ResizeOverlap(
            alpha_scheduler=alpha_scheduler,
            kernel_radius_scheduler=kernel_radius_scheduler,
            algorithm=overlap_algorithm_factory(overlap_algorithm),
        )

        def execute_overlap(context: SamplingCallbackContext):
            # Modifies context.denoised in place
            noise = context.noise
            print("Noise shape", noise.shape)
            print("Step index", context.step_index)
            print("Total steps", context.total_steps)
            frame_seq = [frame.unsqueeze(0) for frame in noise]
            print("frame shape", frame_seq[0].shape)
            estimated_denoising_timestep = 1000 - int(((context.step_index + 1) / context.total_steps) * 1000)
            print("sigma", context.sigma)
            print("Estimated Denoising step", estimated_denoising_timestep)
            overlapped_frame_seq = overlap.__call__(
                frame_seq,
                corr_map=correspondence_map,
                step=context.step_index,
                timestep=estimated_denoising_timestep
            )
            for i, frame in enumerate(overlapped_frame_seq):
                context.noise[i] = frame.squeeze()

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