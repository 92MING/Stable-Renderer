if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    __package__ = 'common_utils.stable_render_utils'

import torch
from einops import rearrange
import torch.nn.functional as F

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Any

from .corr_utils import *
from attr import attrs, attrib
from common_utils.global_utils import is_dev_mode
from common_utils.debug_utils import EngineLogger
from common_utils.math_utils import (
    tensor_group_by_then_average, 
    tensor_group_by_then_set_first_occurance,
    adaptive_instance_normalization
)

if TYPE_CHECKING:
    from comfyUI.types import SamplingCallbackContext, IMAGE, EngineData
    from engine.static.corrmap import UpdateMode
    from comfyUI.comfy.ldm.modules.attention import BasicTransformerBlock


class Corresponder(Protocol):
    '''object that defines the correspond function for treating the related values across frames'''
    
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
                         q_context: torch.Tensor,
                         k_context: torch.Tensor,
                         v_context: torch.Tensor,
                         layer: int)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        This method will be called before an self-attention block is executed.
        It is used for modifying the qkv contexts before the self-attention block.
        
        Args:
            - block: the self-attention block instance
            - engine_data: the baking data containing the packed data from previous rendering frames, e.g id_maps, color_maps, ...
            - q_context: the query context tensor to be passed to the query weight
            - k_context: the key context tensor to be passed to the key weight
            - v_context: the value context tensor to be passed to the value weight
            - layer: the layer index of the current frame. This is for you to specify the layer you want to treat in the current frame,
                    e.g. the self-attention block `BasicTransformerBlock` has 16 layers
                    
        Returns:
            - the modified q_context, k_context, v_context tensors
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
    ignore_obj_mat_id_when_update: bool = attrib(default=False)
    
    def post_atten_inject(self,
                          block: "BasicTransformerBlock",
                          engine_data: "EngineData", 
                          origin_values: torch.Tensor,  # flattened tensor, e.g. (8, 4096, 320)
                          layer: int)->torch.Tensor:
        '''the general correspond function that uses the equal contribution method, 
        i.e. assume each pixel contributes equally to the cell it belongs to.'''
        return origin_values
        if layer not in self.layer_range:
            return origin_values
        
        EngineLogger.info(f'doing post attn inject on layer {layer} with ratio {self.post_attn_inject_ratio}. Value shape: {origin_values.shape}, id shape: {engine_data.id_maps.tensor.shape}')
        
    def finished(self, engine_data: "EngineData", images: "IMAGE"):
        '''
        This method will be called when VAE decode is finished. CorrespondMap will be updated with the new pixel values here.
        `correspond_maps` should be provided in `extra_data` on baking mode.
        '''
        if not self.update_corrmap or images is None or engine_data.id_maps is None:
            return
        
        id_maps = engine_data.id_maps.tensor
        masks = engine_data.id_maps.masks   # here is id_maps.masks, not engine_data.masks
        if masks is None:
            masks = torch.ones(id_maps.shape[:-1], dtype=torch.bool, device=id_maps.device)
        corrmaps = engine_data.correspond_maps
        
        if corrmaps:
            for (spriteID, materialID), corrmap in corrmaps.items():    # but suppose there should be only 1 corrmap in baking mode
                corrmap.update(
                    color_frames=images,
                    id_maps=id_maps,
                    mode=self.update_corrmap_mode,
                    masks=masks,
                    spriteID=spriteID,
                    materialID=materialID,
                    ignore_obj_mat_id = self.ignore_obj_mat_id_when_update,  # ignore the object ID
                    inverse_masks=True  # update the pixels that are not masked
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
    update_corrmap_mode: "UpdateMode" = attrib(default='first')
    '''the mode for updating the correspondence map'''
    pre_attn_inject_num_random_frames: int = attrib(default=1)
    '''the number of random frames to inject before the attention block, disable by entering negative value.'''
    _random_frame_indices: torch.Tensor = attrib(default=None, init=False)
    '''the random frame indices to inject before the attention block, will be initialized after the
    first pre_attn_inject call'''

    post_attn_inject_ratio: float = attrib(default=0.6)
    '''final attn value = cached value * post_attn_inject_ratio + origin value * (1 - post_attn_inject_ratio)'''

    step_finished_inject_ratio: float = attrib(default=0.1)
    '''final attn value = cached value * step_finished_inject_ratio + averaged value * (1 - step_finished_inject_ratio)'''
    step_finished_stop_inject_timestep: int = attrib(default=500) 
    '''the timestep to stop the injection of the averaged value, default is 0, which means always inject the averaged value'''

    def prepare(self, engine_data: "EngineData"):
        pass

    def pre_atten_inject(self, 
                         block: "BasicTransformerBlock", 
                         engine_data: "EngineData",
                         q_context: torch.Tensor,
                         k_context: torch.Tensor,
                         v_context: torch.Tensor,
                         layer: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pre_attn_inject_num_random_frames < 0:
            EngineLogger.debug(f"Skipping pre attn inject for layer {layer}")
            return q_context, k_context, v_context
        
        EngineLogger.debug(f"Layer {layer} pre attn inject")
        EngineLogger.debug(f"Pre inject shapes: q_context={q_context.shape}, k_context={k_context.shape}, v_context={v_context.shape}")

        if self._random_frame_indices is None:
            self._random_frame_indices = torch.randint(1, k_context.shape[0], (self.pre_attn_inject_num_random_frames,))

        # Extract random N frames' k v contexts
        random_k = k_context[self._random_frame_indices]
        random_v = v_context[self._random_frame_indices]

        # Expand and merge the random N frames' k v contexts
        expanded_random_k = torch.cat(
            [k for k in random_k], dim=0).unsqueeze(0).expand(k_context.shape[0], -1, -1)
        expanded_random_v = torch.cat(
            [v for v in random_v], dim=0).unsqueeze(0).expand(v_context.shape[0], -1, -1)

        k_context = expanded_random_k
        v_context = expanded_random_v

        EngineLogger.debug(f"Post inject shapes: q_context={q_context.shape}, k_context={k_context.shape}, v_context={v_context.shape}")
        return q_context, k_context, v_context
    
    def post_atten_inject(self, 
                          block: "BasicTransformerBlock", 
                          engine_data: "EngineData",
                          origin_values: torch.Tensor,
                          layer: int
        ) -> torch.Tensor:
        return origin_values

        EngineLogger.debug(f"Layer {layer} post attn inject")
        # post atten inject does not seem effective
        if layer in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            EngineLogger.debug(f"Skipping post attn inject for layer {layer}")
            return origin_values
        
        # post attention values have shape (b, (h w), c)
        b, hw, c = origin_values.shape
        if pow(hw, 1/2) != int(pow(hw, 1/2)):
            raise ValueError(f"Dimension hw={hw} is not a perfect square.")

        h = w = int(pow(hw, 1/2))
        rearranged_features = rearrange(origin_values, "b (h w) c -> b c h w", h=h, w=w)
        EngineLogger.debug(f'Rearranged features {rearranged_features.shape}')

        id_map = engine_data.id_maps
        vertex_screen_info = id_map.create_vertex_screen_info().to(origin_values.device)

        screen_x_coords = (vertex_screen_info[:, 4] * id_map.width).to(torch.int)
        screen_y_coords = (vertex_screen_info[:, 5] * id_map.height).to(torch.int)
        screen_frame_indices = vertex_screen_info[:, 6].to(torch.int)

        # Upscale to match the size of the id map
        upscaled_features = F.interpolate(rearranged_features, size=(id_map.height, id_map.width), mode='nearest')

        corresponding_features = upscaled_features[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ]

        indexed_corr_features = torch.cat([
            corresponding_features,
            vertex_screen_info[:, 3].unsqueeze(-1)  # vector index
        ], dim=1)

        averaged_features, unique_indices = tensor_group_by_then_average(
            indexed_corr_features,
            index_column=-1,
            value_columns=[i for i in range(c)],
            return_unique=True
        )

        alpha_weighted_average_features = (1 - self.post_attn_inject_ratio) * corresponding_features + \
            self.post_attn_inject_ratio * averaged_features.to(corresponding_features.dtype)
        
        upscaled_features[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ] = alpha_weighted_average_features.to(upscaled_features.dtype)

        downscaled_features = F.interpolate(upscaled_features, size=(h, w), mode='nearest')

        EngineLogger.debug(f"Downscaled features {downscaled_features.shape}")

        normalized_features = adaptive_instance_normalization(
            rearranged_features,
            downscaled_features
        )

        EngineLogger.debug(f"Normalized features {normalized_features.shape}")

        return rearrange(normalized_features, "b c h w -> b (h w) c").contiguous()

    
    def step_finished(self, engine_data: "EngineData", sampling_context: "SamplingCallbackContext"):
        timestep = sampling_context.timestep
        EngineLogger.info(f'Current timestep: {timestep}')
        if timestep < self.step_finished_stop_inject_timestep:
            EngineLogger.info(f"Current timestep: {timestep}, stop injecting the averaged value at timestep {self.step_finished_stop_inject_timestep}.")
            return

        id_map = engine_data.id_maps
        vertex_screen_info = id_map.create_vertex_screen_info()
        
        noise_copy = sampling_context.noise.clone()
        vertex_screen_info = vertex_screen_info.to(noise_copy.device)
        batch, channels, height, width = noise_copy.shape

        screen_x_coords = (vertex_screen_info[:, 4] * width).to(torch.int)
        screen_y_coords = (vertex_screen_info[:, 5] * height).to(torch.int)
        screen_frame_indices = vertex_screen_info[:, 6].to(torch.int)
        
        # (num_vertex_screen_info, channels)
        if noise_copy.dtype == torch.float16:
            noise_copy = noise_copy.to(torch.float32)

        EngineLogger.debug(
            f"Corresponding noises shape: {noise_copy[screen_frame_indices, :, screen_x_coords, screen_y_coords].shape}"
        )

        corresponding_noises = noise_copy[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ]

        indexed_corr_noises = torch.cat([
            corresponding_noises,
            vertex_screen_info[:, 3].unsqueeze(-1)  # vector index
        ], dim=1)
        
        EngineLogger.debug(f"Indexed corr noises shape: {indexed_corr_noises.shape}")

        # (num_vertex_screen_info, channels + 1)
        averaged_noises, unique_indices = tensor_group_by_then_average(
            indexed_corr_noises,
            index_column=-1,
            value_columns=[i for i in range(channels)],
            return_unique=True
        )
        print(indexed_corr_noises)
        print(averaged_noises)
        print(unique_indices)
        EngineLogger.debug(f"Unique indices shape: {unique_indices.shape}")
        EngineLogger.debug(f"Averaged noises shape: {averaged_noises.shape}")

        alpha_weighted_average_noises = (1 - self.step_finished_inject_ratio) * corresponding_noises + \
            self.step_finished_inject_ratio * averaged_noises.to(corresponding_noises.dtype)
        
        noise_copy[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ] = alpha_weighted_average_noises.to(noise_copy.dtype)

        normalized_noise = adaptive_instance_normalization(
            sampling_context.noise.clone(),
            noise_copy
        )

        # mask_out_background = torch.ones_like(normalized_noise)
        # mask_out_background[
        #     screen_frame_indices,
        #     :,  
        #     screen_y_coords,
        #     screen_x_coords,
        # ] = 0
        # normalized_noise = normalized_noise * mask_out_background + noise_copy * (1 - mask_out_background)

        for i, normalized_noise in enumerate(normalized_noise):
            sampling_context.noise[i] = normalized_noise


__all__ = ['Corresponder', 'DefaultCorresponder', 'OverlapCorresponder']


if __name__ == '__main__':
    # test the default corresponder's `finished` method
    import os
    from comfyUI.types import EngineData, CorrespondMaps
    from common_utils.path_utils import RESOURCES_DIR, TEMP_DIR
    from engine.static.corrmap import IDMap, CorrespondMap
    from comfyUI.stable_rendering import LegacyImageSequenceLoader
    import numpy as np
    from PIL import Image
    from pathlib import Path
    
    # data_dir = RESOURCES_DIR / 'example-map-outputs' / 'miku-sphere'
    # data_dir = r'D:\Stable-Renderer\output\runtime_map\2024-05-17_6'.replace('\\', '/')
    data_dir = r'D:\Stable-Renderer\output\runtime_map\2024-05-17_11'.replace('\\', '/')
    data_dir = Path(data_dir)
    color_dir = data_dir / 'color'
    id_dir = data_dir / 'id'
    output_dir = TEMP_DIR
    
    id_maps = IDMap.from_directory(id_dir) 
    first_id_mask = id_maps.masks[0].unsqueeze(-1).squeeze(0)
    first_id_mask = 1 - torch.cat([first_id_mask, ] * 3, dim=-1)
    Image.fromarray((first_id_mask * 255).numpy().astype(np.uint8)).save(output_dir / 'first_id_mask.png')
    
    color_imgs = [os.path.join(color_dir, img) for img in os.listdir(color_dir) if img.endswith('.png')]
    color_maps, color_masks = LegacyImageSequenceLoader()(color_imgs)
    color_masks = color_masks.unsqueeze(-1)
    color_masks = torch.cat([color_masks, ] * 3, dim=-1)
    final_colors = torch.cat([color_maps, 1 - color_masks], dim=-1)
    print('color_maps shape:', color_maps.shape, 'color_masks shape:', color_masks.shape, 'final_colors shape:', final_colors.shape)
    
    corrmap = CorrespondMap(name='miku_k=6', k=6)
    
    engineData = EngineData(
        frame_indices=[i for i in range(len(id_maps))],
        id_maps=id_maps,
        correspond_maps=CorrespondMaps({(1, 0): corrmap})
    )
    
    corresponder = DefaultCorresponder(ignore_obj_mat_id_when_update=True)
    corresponder.finished(engineData, final_colors)
    real_dump_path = corrmap.dump(TEMP_DIR / 'test_corresponder_finished')
    
    corrmap2 = CorrespondMap.Load(real_dump_path, name='test_load')
    print('equal:', torch.all(corrmap2._values == corrmap._values), torch.all(corrmap2._writtens == corrmap._writtens))