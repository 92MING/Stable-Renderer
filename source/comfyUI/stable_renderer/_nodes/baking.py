from typing import Tuple
from comfyUI.types import *


class BakingDataNode(StableRendererNodeBase):
    '''nodes providing the runtime data during baking'''
    
    def __call__(self, )->Tuple[Named(float, 'azimuth'), 
                                Named(float, 'elevation')]:
        pass