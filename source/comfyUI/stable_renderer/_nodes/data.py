from typing import Tuple, TYPE_CHECKING
from comfyUI.types import *

if TYPE_CHECKING:
    from engine.static.texture import Texture 


class BakingDataNode(StableRendererNodeBase):
    '''nodes providing the runtime data during baking'''
    
    def __call__(self, 
                 baking_data: BakingData  # this is hidden value, will be passed during runtime
                )->Tuple[Named(float, 'azimuth'), Named(float, 'elevation')]:
        pass


class FrameDataNode(StableRendererNodeBase):
    '''nodes providing the runtime data during rendering'''
    
    def __call__(self, 
                 frame_data: FrameData  # this is hidden value, will be passed during runtime
                )->Tuple[
                        Named("Texture", "colorMap"),
                        Named("Texture", "idMap"),
                        Named("Texture", "posMap"),
                        Named("Texture", "normalAndDepthMap"),
                        Named("Texture", "noiseMap")
                ]:      # type: ignore
        if frame_data is None:
                return (None, None, None, None, None)
        return frame_data.colorMap, frame_data.idMap, frame_data.posMap, frame_data.normalAndDepthMap, frame_data.noiseMap
    
    
    
__all__ = ['BakingDataNode', 'FrameDataNode']