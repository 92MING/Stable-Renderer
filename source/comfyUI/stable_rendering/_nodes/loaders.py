# Legacy loaders read path in string format
import os
import numpy as np

import torch
from einops import rearrange
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

from comfyUI.types import *
from typing import Literal
from engine.static.corrmap import IDMap

from common_utils.path_utils import extract_index
from common_utils.math_utils import adaptive_instance_normalization, tensor_group_by_then_randn_init
from common_utils.debug_utils import ComfyUILogger


class ImageSequenceLoader(StableRenderingNode):
    Category = "loader"

    def __call__(self,
                 directory: STRING(forceInput=True),  # type: ignore
                 frame_start: INT(min=0) = 0,  # type: ignore
                 num_frames: INT(min=1) = 16,  # type: ignore
                 sd_version: Literal['SD15', "SDXL"] = 'SD15',
    ) -> IMAGE:
        """Load image sequence from a given folder

        Note: The filename of images should contain a number, indicating its frame index in
        a sequence. E.g. depth_0.png, depth_2.png. Undefined behaviour will occur if such indicator
        is not found.

        Args:
            directory (str): The absolute path to the desired folder
            frame_start (int, optional): The first returning frame. Defaults to 0.
            num_frames (int, optional): Number of frames to be returned. Defaults to 16.
            sd_version (Literal, option): Version of SD, either 'SD15 or SDXL', which determines the
                output size of image

        Returns:
            torch.Tensor: Tensor of shape (num_frames, height, width, channels)
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found")
        if frame_start < 0:
            raise ValueError("frame_start takes value larger than or equal to 0, got ", frame_start)
        if num_frames <= 0:
            raise ValueError("num_frames takes value larger 0, got ", frame_start)
        if sd_version not in ["SD15", "SDXL"]:
            raise ValueError("sd_version should be either SD15 or SDXL")

        # Extract and sort image files
        file_filter = lambda fname: fname.endswith((".jpeg", ".png", ".bmp", ".jpg")) \
            and os.path.exists(os.path.join(directory, fname))
        filenames = list(filter(file_filter, os.listdir(directory)))
        reordered_filenames = sorted(
            filenames, key=lambda x: extract_index(x, filenames.index(x)))

        # Read images as tensor from filenames
        tensor_images = []
        target_size = (512, 512) if sd_version == "SD15" else (1024, 1024)

        for filename in reordered_filenames[frame_start: frame_start+num_frames]:
            img_path = os.path.join(directory, filename)
            tensor_img = read_image(img_path, mode=ImageReadMode.RGB)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_img = F.interpolate(tensor_img, size=target_size)
            tensor_img = tensor_img.permute(0, 2, 3, 1) / 255.0
            tensor_images.append(tensor_img)

        if not tensor_images:
            return None # type: ignore
        
        ComfyUILogger.debug(f"Loaded image sequence with shape {tensor_images[0].shape}")
        return torch.cat(tensor_images, dim=0) 


class NoiseSequenceLoader(StableRenderingNode):
    Category = "loader"
    
    def __call__(self, 
                 directory: STRING(forceInput=True),  # type: ignore
                 frame_start: INT(min=0) = 0,  # type: ignore
                 num_frames: INT(min=1) = 16,  # type: ignore
                 sd_version: Literal['SD15', "SDXL"] = 'SD15',
                 ) -> LATENT:
        '''load sequence of noise from img or directly from npy.'''
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found")
        if frame_start < 0:
            raise ValueError(
                "frame_start takes value larger than or equal to 0, got ", frame_start)
        if num_frames <= 0:
            raise ValueError("num_frames takes value larger 0, got ", frame_start)
        if sd_version not in ["SD15", "SDXL"]:
            raise ValueError("sd_version should be either SD15 or SDXL")

        # Extract and sort noise npy file paths
        file_filter = lambda fname: fname.endswith((".jpeg", ".png", ".bmp", ".jpg", ".npy")) \
            and os.path.exists(os.path.join(directory, fname))
        filenames = list(filter(file_filter, os.listdir(directory)))
        reordered_filenames = sorted(
            filenames,key=lambda x: extract_index(x, filenames.index(x)))

        # Read npy files as tensor
        tensors = []
        for filename in reordered_filenames[frame_start: frame_start+num_frames]:
            data_path = os.path.join(directory, filename)
            if data_path.endswith('.npy'):
                tensor = torch.from_numpy(np.load(data_path)).squeeze()
                if not len(tensor.shape) == 3:
                    raise ValueError(f"Invalid shape of noise tensor: {tensor.shape}.")
                if not (tensor.shape[-1] == 4 or tensor.shape[1] == 4): # not chw or hwc
                    raise ValueError(f"Invalid noise tensor shape: {tensor.shape}.")
            else:
                tensor = read_image(data_path, mode=ImageReadMode.RGB_ALPHA) / 255.0
            tensors.append(tensor)

        if len(tensors) == 0:
            return None

        if any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError(f"Tensor data has inconsistent shapes.")

        # Processing noise
        noise = torch.stack(tensors, dim=0)
        _, height, width, channel = noise.shape
        assert channel == 4, "Noise shape should be in BHW4"
        ComfyUILogger.debug(f"Loaded noise with shape {noise.shape}")

        reshape_magnitude = -1
        if sd_version == "SD15":
            if height % 64 != 0 or width % 64 != 0:
                raise ValueError("Noise shape for SD15 should be divisible by 64")
            reshape_magnitude = height // 64
        elif sd_version == "SDXL":
            if height % 128 != 0 or width % 128 != 0:
                raise ValueError("Noise shape for SDXL should be divisible by 128")
            reshape_magnitude = height // 128
        assert reshape_magnitude != -1

        style_feat_tensor = noise.clone()
        noise = noise.view(-1, reshape_magnitude, reshape_magnitude, 4).mean(dim=(1, 2))
        noise = noise.view(-1, height//reshape_magnitude, width//reshape_magnitude, 4)
        noise = adaptive_instance_normalization(
            noise, style_feat_tensor, mode='NHWC')  # here the shape changed to 1,4,h//8,w//8

        ComfyUILogger.debug(
            f"Noise shape after adaptive_instance_normalization {noise.shape}")
        return LATENT(samples=torch.zeros_like(noise), noise=noise)


class CreateNoiseSequenceFromIdMap(StableRenderingNode):
    Category = "loader"

    def __call__(self,
                 id_map: IDMap,  # type: ignore
                 seed: INT(0, 0xffffffffffffffff),  # type: ignore
                 sd_version: Literal['SD15', "SDXL"] = 'SD15',
                 downsample_option: Literal['mean', 'max', 'min', 'nearest'] = 'nearest',
    ) -> LATENT:
        '''Create noise sequence from ID map.'''
        if sd_version not in ["SD15", "SDXL"]:
            raise ValueError("sd_version should be either SD15 or SDXL")
        if downsample_option not in ["mean", "max", "min", "nearest"]:
            raise ValueError("downsample_option should be either mean, max, min, or nearest")
        if not id_map:
            raise ValueError("ID map is empty.")

        # Initialize noise tensor
        latent_generator = torch.manual_seed(seed)
        noise_generator = torch.manual_seed(seed + 1)

        if sd_version == "SD15":
            height, width = 512, 512
        elif sd_version == "SDXL":
            height, width = 1024, 1024
        else:
            assert False
        
        latent = torch.randn([1, 4, height, width],
                             device="cpu",  # generator need to run on cpu
                             generator=latent_generator)
        latent = latent.repeat(id_map.frame_count, 1, 1, 1).to('cuda')

        noise = torch.randn([1, 4, height, width],
                            device="cpu",
                            generator=noise_generator)
        noise = noise.repeat(id_map.frame_count, 1, 1, 1).to('cuda')

        # Update latent tensor with vertex screen info
        vertex_screen_info = id_map.create_vertex_screen_info().to('cuda')

        screen_x_coords = (vertex_screen_info[:, 4] * width).long()
        screen_y_coords = (vertex_screen_info[:, 5] * height).long()
        screen_frame_indices = vertex_screen_info[:, 6].long()

        corresponding_latents = latent[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords
        ]

        indexed_corresponding_latents = torch.cat([
                corresponding_latents,
                vertex_screen_info[:, 3].unsqueeze(-1)
        ], dim=-1)

        randn_init_latents, _ = tensor_group_by_then_randn_init(
            indexed_corresponding_latents,
            index_column=-1,
            value_columns=[i for i in range(4)],
            return_unique=True
        )

        latent[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ] = randn_init_latents

        # Update noise tensor with vertex screen info
        corresponding_noises = noise[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords
        ]

        indexed_corresponding_noises = torch.cat([
            corresponding_noises,
            vertex_screen_info[:, 3].unsqueeze(-1)
        ], dim=-1)

        randn_init_noises, _ = tensor_group_by_then_randn_init(
            indexed_corresponding_noises,
            index_column=-1,
            value_columns=[i for i in range(4)],
            return_unique=True
        )

        noise[
            screen_frame_indices,
            :,
            screen_y_coords,
            screen_x_coords,
        ] = randn_init_noises

        if downsample_option == "nearest":
            latent = F.interpolate(latent, size=(height//8, width//8), mode="nearest")
            noise = F.interpolate(noise, size=(height//8, width//8), mode="nearest")
            return LATENT(samples=latent, noise=noise)

        if downsample_option == "mean":
            latent = latent.view(-1, 4, 8, 8).mean(dim=(1, 2))
            noise = noise.view(-1, 4, 8, 8).mean(dim=(1, 2))
        elif downsample_option == "max":
            latent = latent.view(-1, 4, 8, 8).amax(dim=(1, 2))
            noise = noise.view(-1, 4, 8, 8).amax(dim=(1, 2))
        elif downsample_option == "min":
            latent = latent.view(-1, 4, 8, 8).amin(dim=(1, 2))
            noise = noise.view(-1, 4, 8, 8).amin(dim=(1, 2))
        else:
            assert False
        latent = latent.view(-1, 4, height//8, width//8)
        noise = noise.view(-1, 4, height//8, width//8)

        return LATENT(samples=torch.zeros_like(noise), noise=noise)


class CreateIdenticalNoiseSequence(StableRenderingNode):
    Category = "loader"

    def __call__(self,
                 seed: INT(0, 0xffffffffffffffff),  # type: ignore
                 num_frames: INT(0, 100000), # type: ignore
                 sd_version: Literal['SD15', "SDXL"] = 'SD15',
    ) -> LATENT:
        '''Create noise sequence from ID map.'''
        if sd_version not in ["SD15", "SDXL"]:
            raise ValueError("sd_version should be either SD15 or SDXL")
        if num_frames <= 0:
            raise ValueError("num_frames should be larger than 0.")

        # Initialize noise tensor
        latent_generator = torch.manual_seed(seed)
        noise_generator = torch.manual_seed(seed + 1)

        if sd_version == "SD15":
            height, width = 64, 64
        elif sd_version == "SDXL":
            height, width = 128, 128
        else:
            assert False
        
        latent = torch.randn([1, 4, height, width],
                             device="cpu",  # generator need to run on cpu
                             generator=latent_generator)
        latent = latent.repeat(num_frames, 1, 1, 1).to('cuda')

        noise = torch.randn([1, 4, height, width],
                            device="cpu",
                            generator=noise_generator)
        noise = noise.repeat(num_frames, 1, 1, 1).to('cuda')

        return LATENT(samples=latent, noise=noise)
    
    
class IDSequenceLoader(StableRenderingNode):
    Category = "loader"

    def __call__(self,
                 directory: STRING(forceInput=True),  # type: ignore
                 frame_start: INT(min=0) = 0,  # type: ignore
                 num_frames: INT(min=1) = 16,  # type: ignore
    ) -> IDMap:
        '''Load ID files from img or directly from npy.'''
        return IDMap.from_directory(
            directory=directory, 
            frame_start=frame_start,
            num_frames=num_frames,
            use_frame_indices_from_filename=False
        )


__all__ = ["ImageSequenceLoader", "NoiseSequenceLoader", "CreateNoiseSequenceFromIdMap", "CreateIdenticalNoiseSequence", "IDSequenceLoader"]
