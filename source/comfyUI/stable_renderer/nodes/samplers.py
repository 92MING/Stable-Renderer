from comfyUI.types import *
from .node_base import StableRendererNodeBase
from comfyUI.nodes import common_ksampler


_default_sampler = COMFY_SAMPLERS.__args__[0]
_default_scheduler = COMFY_SCHEDULERS.__args__[0]


class StableRenderSampler(StableRendererNodeBase):
    
    Category = "sampling"

    def __call__(self, 
                 model: MODEL,
                 positive: "CONDITIONING",
                 negative: "CONDITIONING",
                 latent_image: LATENT,
                 add_noise: bool= False,
                 noise_seed: INT(0, 0xffffffffffffffff)=0,
                 steps: INT(1, 10000)=20,
                 cfg: FLOAT(0.0, 100.0, 0.01, round=0.01)=8.0,
                 sampler_name: COMFY_SAMPLERS=_default_sampler,
                 scheduler: COMFY_SCHEDULERS=_default_scheduler,
                 start_at_step: INT(0, 10000)=0,
                 end_at_step: INT(0, 10000)=10000,
                 return_with_leftover_noise: bool=False,
                 denoise=1.0)->LATENT:
        
        force_full_denoise = not return_with_leftover_noise
        disable_noise = not add_noise
        return common_ksampler(model=model, 
                               seed=noise_seed, 
                               steps=steps, 
                               cfg=cfg, 
                               sampler_name=sampler_name, 
                               scheduler=scheduler, 
                               positive=positive, 
                               negative=negative, 
                               latent=latent_image, 
                               denoise=denoise, 
                               disable_noise=disable_noise, 
                               start_step=start_at_step, 
                               last_step=end_at_step, 
                               force_full_denoise=force_full_denoise)   # type: ignore