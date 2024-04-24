from typing import Tuple
from comfyUI.types import *


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
                ):
        pass
    
    
__all__ = ['BakingDataNode', 'FrameDataNode']