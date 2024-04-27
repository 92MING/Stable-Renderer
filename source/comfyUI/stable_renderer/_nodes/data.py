from typing import Tuple, TYPE_CHECKING
from comfyUI.types import *
from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import is_dev_mode, is_verbose_mode
if TYPE_CHECKING:
    from comfyUI.stable_renderer import IDMap


class BakingDataNode(StableRendererNodeBase):
    '''nodes providing the runtime data during baking'''
    
    def __call__(self, 
                 baking_data: BakingData  # this is hidden value, will be passed during runtime
                )->Tuple[
                        Named(float, 'azimuth'),        # type: ignore
                        Named(float, 'elevation')       # type: ignore
                ]:
        pass    # TODO


class FrameDataNode(StableRendererNodeBase):
    '''nodes providing the runtime data during rendering'''
    
    def __call__(self, 
                 frame_data: FrameData  # this is hidden value, will be passed during runtime
                )->Tuple[
                        Named(IMAGE, "color"),  # type: ignore
                        Named("IDMap", "id"),     # type: ignore
                        Named(IMAGE, "pos"),    # type: ignore
                        Named(IMAGE, "normal"), # type: ignore
                        Named(IMAGE, "depth"),  # type: ignore
                        Named(LATENT, "noise"), # type: ignore
                        Named(MASK, "mask")     # type: ignore
                ]:
        
        if frame_data is None:
                return (None, None, None, None, None, None, None)
        
        return (frame_data.color_map, 
                frame_data.id_map, 
                frame_data.pos_map, 
                frame_data.normal_map,
                frame_data.depth_map, 
                frame_data.noise_map,
                frame_data.mask)


class InferenceOutputNode(StableRendererNodeBase):
    '''the node type for defining output data which will be passed to `RenderManager` later.'''
    
    IsOutputNode = True
    '''this node is an output node(though it has no UI output).'''
    Unique = True
    '''only one instance of this node type is allowed in the graph.'''
    
    def __call__(self, 
                 colorImg: IMAGE, 
                 context: InferenceContext,      # this is hidden value, will be passed during runtime
                 )->InferenceOutput:
        out = InferenceOutput(colorImg)
        context.final_output = out
        return out
    
    # this will be called instead of `__call__` when web server is running
    def __server_call__(self,
                        colorImg: IMAGE,                # required type
                        context: InferenceContext,      # hidden type
                        prompt: PROMPT,                 # hidden type
                        png_info: EXTRA_PNG_INFO,       # hidden type
                        save: bool=False,               # optional type
                        ) -> UIImage:
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug('InferenceOutputNode.__server_call__')
        context.final_output = InferenceOutput(colorImg)
        return UIImage(colorImg, type='temp' if not save else 'output', prompt=prompt, png_info=png_info)
        
    
    
__all__ = ['BakingDataNode', 'FrameDataNode', 'InferenceOutputNode']