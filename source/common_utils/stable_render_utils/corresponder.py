import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Any
from .corr_utils import cells_value_overlap
from attr import attrs, attrib
from common_utils.global_utils import is_dev_mode
from common_utils.debug_utils import EngineLogger

if TYPE_CHECKING:
    from comfyUI.types import SamplingCallbackContext, IMAGE, EngineData
    from engine.static.corrmap import UpdateMode
    from comfyUI.comfy.ldm.modules.attention import BasicTransformerBlock


class Corresponder(Protocol):
    '''object that defines the correspond function for treating the related values across frames'''
    layer_range: tuple[int, ...]
    '''
    Define the layers that the correspond function will be applied to.
    Default is the 6th layer(the middle layer of the whole unet).
    '''
    update_corrmap: bool
    '''whether to update the correspondence map'''
    update_corrmap_mode: "UpdateMode"
    '''the mode for updating the correspondence map'''
    post_attn_inject_ratio: float
    '''final attn value = cached value * post_attn_inject_ratio + origin value * (1 - post_attn_inject_ratio)'''
    
    @abstractmethod
    def prepare(self, engine_data: "EngineData"):
        '''
        This method will be called before the baking inference starts.
        
        Engine data is a pack of frame datas, including colors, normals, ids, etc.
        You can directly modify it for any preparation purpose.
        '''
    
    @abstractmethod
    def pre_atten_inject(self, 
                         block: "BasicTransformerBlock",
                         engine_data: "EngineData",
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         layer: int)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        This method will be called before an self-attention block is executed.
        It is used for modifying the QKV values before the self-attention block.
        
        Args:
            - block: the self-attention block instance
            - engine_data: the baking data containing the packed data from previous rendering frames, e.g id_maps, color_maps, ...
            - q: the query tensor
            - k: the key tensor
            - v: the value tensor
            - layer: the layer index of the current frame. This is for you to specify the layer you want to treat in the current frame,
                    e.g. the self-attention block `BasicTransformerBlock` has 16 layers
                    
        Returns:
            - the modified q, k, v tensors
        '''
    
    @abstractmethod
    def post_atten_inject(self,
                          block: "BasicTransformerBlock", 
                          engine_data: "EngineData",
                          origin_values: torch.Tensor,
                          layer: int)->torch.Tensor:
        '''
        This method will be called after an self-attention block is executed.
        
        Args:
            - block: the self-attention block instance
            - engine_data: the baking data containing the packed data from previous rendering frames, e.g id_maps, color_maps, ...
            - origin_values: the attention values output from the Transformer block (flattened tensor)
            - layer: the layer index of the current frame. This is for you to specify the layer you want to treat in the current frame,
                    e.g. the self-attention block `BasicTransformerBlock` has 16 layers
                    
        Returns:
            - the modified frame_values tensor
        '''

    @abstractmethod
    def step_finished(self, engine_data: "EngineData", sampling_context: "SamplingCallbackContext"):
        '''
        This method will be called when a timestep is finished.
        You could modify latent directly, or save some values here. They could be found in `sampling_context`.
        '''

    @abstractmethod
    def finished(self, engine_data: "EngineData", images: "IMAGE"):
        '''
        This method will be called when VAE decode is finished.
        You can use it to save baking results.
        '''

@attrs
class DefaultCorresponder:
    '''The default implementation of `Corresponder`'''
    
    layer_range: tuple[int, ...] = attrib(default=(6,))
    '''
    Define the layers that the correspond function will be applied to.
    Default is the 6th layer(the middle layer of the whole unet).
    '''
    update_corrmap: bool = attrib(default=True)
    '''whether to update the correspondence map'''
    update_corrmap_mode: "UpdateMode" = attrib(default='first_avg')
    '''the mode for updating the correspondence map'''
    post_attn_inject_ratio: float = attrib(default=0.6)
    '''final attn value = cached value * post_attn_inject_ratio + origin value * (1 - post_attn_inject_ratio)'''
    
    def post_atten_inject(self,
                          block: "BasicTransformerBlock",
                          engine_data: "EngineData", 
                          origin_values: torch.Tensor,  # flattened tensor
                          layer: int)->torch.Tensor:
        '''the general correspond function that uses the equal contribution method, 
        i.e. assume each pixel contributes equally to the cell it belongs to.'''
        return origin_values
        if layer not in self.layer_range:
            return origin_values
        
        EngineLogger.info(f'doing post attn inject on layer {layer} with ratio {self.post_attn_inject_ratio}')
            
        flatten_id_maps = engine_data.id_maps.tensor.clone()
        if flatten_id_maps.dim() == 4:
            flatten_id_maps = flatten_id_maps.view(engine_data.frame_count, -1, 4).contiguous()
        new_values = torch.zeros_like(origin_values)
        pixels_per_cell = flatten_id_maps.shape[1] // origin_values.shape[1] # e.g. (512*512) // (64*64) = 64, means 64 pixels in a cell
        contributions = torch.ones_like(flatten_id_maps[..., 0]).float() / pixels_per_cell
        
        cells_value_overlap(flatten_id_maps, origin_values, new_values, contributions)
        new_values = self.post_attn_inject_ratio * new_values + (1 - self.post_attn_inject_ratio) * origin_values
        return new_values
    
    def finished(self, engine_data: "EngineData", images: "IMAGE"):
        '''
        This method will be called when VAE decode is finished. CorrespondMap will be updated with the new pixel values here.
        `correspond_maps` should be provided in `extra_data` on baking mode.
        '''
        if not self.update_corrmap or images is None or engine_data.id_maps is None:
            return
        exit()
        id_maps = engine_data.id_maps
        final_colors = images
        masks = engine_data.masks
        corrmaps = engine_data.correspond_maps
        
        if corrmaps:
            for (spriteID, materialID), corrmap in corrmaps.items():    # but suppose there should be only 1 corrmap in baking mode
                if masks is not None:
                    this_masks = masks.clone()
                    if spriteID is not None:
                        this_masks[:, id_maps.tensor[0, ...] != spriteID] = 1 # 1 means should not include
                    if materialID is not None:
                        this_masks[:, id_maps.tensor[0, ...] != materialID] = 1
                else:
                    this_masks = None
                
                corrmap.update(
                    color_frames=final_colors.clone(),
                    id_maps=id_maps,
                    mode=self.update_corrmap_mode,
                    masks=this_masks
                )
    

@attrs
class OverlapCorresponder:
    """Implements the Corresponder interface"""
    '''The default implementation of `Corresponder`'''
    
    layer_range: tuple[int, ...] = attrib(default=(6,))
    '''
    Define the layers that the correspond function will be applied to.
    Default is the 6th layer(the middle layer of the whole unet).
    '''
    update_corrmap: bool = attrib(default=True)
    '''whether to update the correspondence map'''
    update_corrmap_mode: "UpdateMode" = attrib(default='first_avg')
    '''the mode for updating the correspondence map'''
    post_attn_inject_ratio: float = attrib(default=0.6)
    '''final attn value = cached value * post_attn_inject_ratio + origin value * (1 - post_attn_inject_ratio)'''

    def prepare(self, engine_data: "EngineData"):
        pass

    def pre_atten_inject(self, 
                         block: "BasicTransformerBlock", 
                         engine_data: "EngineData",
                         q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         layer: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return q, k, v
    
    def post_atten_inject(self, 
                          block: "BasicTransformerBlock", 
                          engine_data: "EngineData",
                          origin_values: torch.Tensor,
                          layer: int
        ) -> torch.Tensor:
        return origin_values
    
    def step_finished(self, engine_data: "EngineData", sampling_context: "SamplingCallbackContext"):
        pass

    
__all__ = ['Corresponder', 'DefaultCorresponder', 'OverlapCorresponder']