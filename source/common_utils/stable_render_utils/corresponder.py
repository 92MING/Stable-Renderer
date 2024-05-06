import numpy
import torch

from attr import attrs, attrib
from typing import Optional, TYPE_CHECKING, Callable, TypeAlias, Literal

if TYPE_CHECKING:
    from engine.static.texture import Texture
    from comfyUI.types import BakingData

@attrs
class IDMap:
    '''IDMap represents the ID information of each frame, which is used to build the correspondence map(a packed version of IDMap)'''
    
    frame_index: int = attrib()
    '''the frame index of this map'''

    _origin_tex: Optional["Texture"] = attrib(alias='origin_tex')
    '''the original texture containing the ID information'''
    
    _tensor: Optional[torch.Tensor] = attrib(default=None, alias='_tensor')
    _ndarray: Optional[numpy.ndarray] = attrib(default=None, alias='_ndarray')
    
    @property
    def tensor(self)->torch.Tensor:
        '''get the info data in tensor format'''
        if self._tensor is None:
            if not self._origin_tex:
                raise ValueError("origin_tex of IDMap is not set, cannot get tensor data")
            self._tensor = self._origin_tex.tensor(update=True, flip=True)
        return self._tensor
    
    @property
    def ndarray(self)->numpy.ndarray:
        '''get the info data in numpy array format'''
        if self._ndarray is None:
            self._ndarray = self.tensor.cpu().numpy()
        return self._ndarray
    
    def __deepcopy__(self):
        tensor = self.tensor.clone()
        ndarray = self._ndarray
        if ndarray is not None:
            ndarray = ndarray.copy()
        m = IDMap(frame_index=self.frame_index, origin_tex=None, _tensor=tensor, _ndarray=ndarray)
        return m


CorrespondStage:TypeAlias = Literal['unet_downscale', 'unet_middle', 'unet_upscale', 'vae_decode']

Corresponder: TypeAlias = Callable[["BakingData", torch.Tensor, CorrespondStage, int], torch.Tensor]

def equal_contrib_corresponder(baking_data: "BakingData", 
                               frame_values: torch.Tensor, 
                               stage: CorrespondStage, 
                               layer: int)->torch.Tensor:
    ...



    
__all__ = ['IDMap', 'Corresponder', 'CorrespondStage', 'equal_contrib_corresponder']