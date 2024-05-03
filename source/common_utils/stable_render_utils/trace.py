import numpy
import torch

from attr import attrs, attrib
from typing import Optional, TYPE_CHECKING, Callable, TypeAlias, Any

if TYPE_CHECKING:
    from engine.static.texture import Texture
    from .corrmap import CorrespondenceMap
    from comfyUI.types import InferenceContext, SamplingCallbackContext

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

class CorrTraceContext:
    '''
    Context of correspondence tracing. For finding the latent relation by correspondence map, and edit the result directly.
    For each timestep in sampling process, the same CorrTraceContext obj will be passed to you(but some fields may change).
    '''
    
    inference_context: "InferenceContext"
    '''context of current inference. During the whole tracing, the inference context will not change.'''
    sampling_context: "SamplingCallbackContext"
    '''
    Context of current sampling. Each timestep will have a different sampling context.
    You should edit the latent image in this context.
    '''
    correspondence_map: "CorrespondenceMap"
    '''the correspondence map that contains each baked pixels's information'''
    
    @property
    def baking_data(self):
        '''
        Alias of `inference_context.baking_data`.
        It is an extra runtime data for baking. For each tracing process, multiple FrameData will be packed together and forms this baking data.
        
        Note: as said above, you should not access the `inference_context.frame_data`, since it is not `FrameData` but `BakingData`.
        '''
        return self.inference_context.baking_data



CorrespondingTracer: TypeAlias = Callable[[CorrTraceContext], Any]



__all__ = ['IDMap']