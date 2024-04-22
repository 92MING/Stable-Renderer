import torch
from comfyUI.types import *
from comfyUI.nodes import custom_ksampler


_default_sampler = COMFY_SAMPLERS.__args__[0]   # type: ignore
_default_scheduler = COMFY_SCHEDULERS.__args__[0]   # type: ignore


class StableRenderSampler(StableRendererNodeBase):
    
    Category = "sampling"

    def __call__(self, 
                 model: MODEL,
                 positive: "CONDITIONING", # type: ignore
                 negative: "CONDITIONING", # type: ignore
                 latent_image: LATENT,
                 noise_seed: INT(0, 0xffffffffffffffff)=0, # type: ignore
                 steps: INT(1, 10000)=20, # type: ignore
                 cfg: FLOAT(0.0, 100.0, 0.01, round=0.01)=8.0, # type: ignore
                 sampler_name: COMFY_SAMPLERS=_default_sampler,
                 scheduler: COMFY_SCHEDULERS=_default_scheduler,
                 denoise: FLOAT(0, 1) = 1.0)->LATENT:
        callbacks = [
            lambda a: print("test 1"),
            lambda b: print("test 2"),
        ]

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
                               callbacks=callbacks)