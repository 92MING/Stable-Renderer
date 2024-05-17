from typing import Optional
from functools import partial

from comfyUI.types import *
from comfyUI.nodes import custom_ksampler
from common_utils.type_utils import is_empty_method
from common_utils.stable_render_utils import (
    Corresponder,
    DefaultCorresponder as _DefaultCorresponder,
    OverlapCorresponder as _OverlapCorresponder,
)
from common_utils.global_utils import is_dev_mode, is_engine_looping
from common_utils.debug_utils import ComfyUILogger
from engine.static import UpdateMode, IDMap

_default_sampler = COMFY_SAMPLERS.__args__[0]   # type: ignore
_default_scheduler = COMFY_SCHEDULERS.__args__[0]   # type: ignore


class DefaultCorresponder(StableRenderingNode):
    
    Category = "sampling"

    def __call__(self, 
                 engine_data: EngineData,   # this is hidden value, will hide on UI
                 update_corrmap: bool=True, 
                 update_mode: UpdateMode = 'first_avg',
                 post_attn_inject_ratio: float = 0.6,
                 )->tuple[
                     Corresponder,
                     VAEDecodeCallback
                 ]:
        '''
        Create the general corresponder that uses the equal contribution method, 
        i.e. assume each pixel contributes equally to the cell it belongs to when calculating overlapping ratio.
        
        Args:
            - engine_data: The current engine data, which packs several frame's data together.
            - update_corrmap: Whether to update the correspondence map. Defaults to True.
            - update_mode: The mode for updating the correspondence map. Defaults to 'first_avg'.
                            possible choices:
                                - replace: 
                                    replace the old value with the new value
                                - replace_avg: 
                                    average the old value with the new value, and put back
                                - first: 
                                    only update the value if the old cell in not written
                                - first_avg: 
                                    only update the value if the old cell in not written. 
                                    But if there are multiple values for the same cell in this update process, average them.
            - post_attn_inject_ratio: The ratio for injecting post-attention values. Defaults to 0.6.
        '''
        corresponder = _DefaultCorresponder(update_corrmap=update_corrmap, 
                                            update_corrmap_mode=update_mode,
                                            post_attn_inject_ratio=post_attn_inject_ratio)
        
        if hasattr(corresponder, "finished") and not is_empty_method(corresponder.finished):
            if is_dev_mode() and is_engine_looping():
                def vae_callback(images: "IMAGE"):
                    UIImage(images, force_saving=True, animated=True)  
                    # for debug, since the output is not saved in the engine loop
                    # this will save the gifs to the output folder
                    corresponder.finished(engine_data, images)
            else:
                vae_callback = partial(corresponder.finished, engine_data)
        else:
            vae_callback = lambda *args, **kwargs: None # do nothing
        return corresponder, vae_callback   # type: ignore


class OverlapCorresponder(StableRenderingNode):
    
    Category = "sampling"

    def __call__(self, 
                 engine_data: EngineData,   # this is hidden value, will hide on UI
                 update_corrmap: bool=True, 
                 update_mode: UpdateMode = 'first_avg',
                 post_attn_inject_ratio: float = 0.6,
                 )->tuple[
                     Corresponder,
                     VAEDecodeCallback
                 ]:
        '''
        Create the general corresponder that uses the equal contribution method, 
        i.e. assume each pixel contributes equally to the cell it belongs to when calculating overlapping ratio.
        
        Args:
            - engine_data: The current engine data, which packs several frame's data together.
            - update_corrmap: Whether to update the correspondence map. Defaults to True.
            - update_mode: The mode for updating the correspondence map. Defaults to 'first_avg'.
                            possible choices:
                                - replace: 
                                    replace the old value with the new value
                                - replace_avg: 
                                    average the old value with the new value, and put back
                                - first: 
                                    only update the value if the old cell in not written
                                - first_avg: 
                                    only update the value if the old cell in not written. 
                                    But if there are multiple values for the same cell in this update process, average them.
            - post_attn_inject_ratio: The ratio for injecting post-attention values. Defaults to 0.6.
        '''
        corresponder = _OverlapCorresponder(update_corrmap=update_corrmap, 
                                            update_corrmap_mode=update_mode,
                                            post_attn_inject_ratio=post_attn_inject_ratio)
        
        if hasattr(corresponder, "finished") and not is_empty_method(corresponder.finished):
            if is_dev_mode() and is_engine_looping():
                def vae_callback(images: "IMAGE"):
                    UIImage(images, force_saving=True, animated=True)  
                    # for debug, since the output is not saved in the engine loop
                    # this will save the gifs to the output folder
                    corresponder.finished(engine_data, images)
            else:
                vae_callback = partial(corresponder.finished, engine_data)
        else:
            vae_callback = lambda *args, **kwargs: None # do nothing
        return corresponder, vae_callback   # type: ignore


class CorrespondSampler(StableRenderingNode):
    
    Category = "sampling"

    def __call__(self, 
                 model: MODEL,
                 positive: "CONDITIONING",
                 negative: "CONDITIONING", 
                 corresponder: Corresponder,
                 engine_data: EngineData,   # this is hidden value, will hide on UI
                 latent: Optional[LATENT] = None, # if none, data will comes from `engine_data.noise_maps`
                 steps: INT(1, 10000)=20, # type: ignore
                 cfg: FLOAT(0.0, 100.0, 0.01, round=0.01)=8.0, # type: ignore
                 sampler_name: COMFY_SAMPLERS=_default_sampler,
                 scheduler: COMFY_SCHEDULERS=_default_scheduler,
                 denoise: FLOAT(0, 1) = 1.0 # type: ignore
                 )->LATENT:
        """
        This sampler is specific for baking process. Latents are not required(since it will be passed through `baking_data`)

        Args:
            model: The model to render.
            positive: The positive conditioning.
            negative: The negative conditioning. This is optional.
            corresponder: The corresponder for stable-rendering
            latent: The latent image. When None, it will be taken from `engine_data.noise_maps`.
            steps: The number of steps. Defaults to 20.
            cfg: The cfg value. Defaults to 8.0.
            sampler_name: The sampler name. Default = euler.
            scheduler: The scheduler. Default = euler.
            denoise: The denoise value. Default = 1.0.

        Returns:
            LATENT: The sampled latent image.
        """
        if hasattr(corresponder, 'prepare') and not is_empty_method(corresponder.prepare):
            corresponder.prepare(engine_data)
        
        if hasattr(corresponder, "step_finished") and not is_empty_method(corresponder.step_finished):
            def on_1_step_finished(engine_data, context: SamplingCallbackContext):
                corresponder.step_finished(engine_data, context)
            callback = [partial(on_1_step_finished, engine_data),]
        else:
            callback = []
        if is_dev_mode() and is_engine_looping():
            def print_progress(context: SamplingCallbackContext):
                ComfyUILogger.info(f"Step {context.step_index+1}/{context.total_steps} finished.")
            callback.append(print_progress) # type: ignore
        

        if latent is None:
            if engine_data is not None:
                latent = engine_data.noise_maps
            else:
                raise ValueError("Input latent is None and engine_data is also None.")

        return custom_ksampler(model=model,
                               seed=None,
                               steps=steps,
                               cfg=cfg,
                               sampler_name=sampler_name,
                               scheduler=scheduler,
                               positive=positive,
                               negative=negative,
                               latent=latent,   # type: ignore
                               denoise=denoise,
                               noise_option='incoming',
                               #noise_option='random',
                               engine_data=engine_data, # kwargs for attention layers to use
                               corresponder=corresponder,   # kwargs for attention layers to use
                               callbacks=callback)[0]   # type: ignore


__all__ = ['DefaultCorresponder', 'CorrespondSampler']