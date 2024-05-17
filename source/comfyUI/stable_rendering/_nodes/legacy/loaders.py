import os
import torch
import numpy as np

from pathlib import Path
from typing import List, Literal
from torchvision.io import read_image, ImageReadMode
from einops import rearrange
from comfyUI.types import *
from engine.static.corrmap import IDMap


class LegacyImageSequenceLoader(StableRenderingNode):
    
    Category = "legacy_loader"
    
    def __call__(self, 
                 imgs: List[PATH(accept_folder=False, 
                                 accept_multiple=True, 
                                 to_folder='temp',
                                 accept_types=[".jpeg", ".png", ".bmp", ".jpg"])],    # type: ignore
                 ) -> tuple[IMAGE, MASK]:
        """Load image sequence from a given folder

        Note: The filename of images should contain a number, indicating its frame index in
        a sequence. E.g. depth_0.png, depth_2.png. Undefined behaviour will occur if such indicator
        is not found.

        Args:
            - imgs: the path to the folder containing the image sequence
        Returns:
            IMAGE: a tensor of shape (num_of_imgs, C, H, W), which concatenates all images in the sequence
        """
        def extract_index(img_path, i):
            img = os.path.basename(img_path)
            if img.split('.')[0].split('_')[-1].isdigit():  # e.g. depth_0.png
                return int(img.split('.')[0].split('_')[-1])
            elif img.split('_')[0].isdigit():   # e.g. 0_depth.png
                return int(img.split('_')[0])
            return i    # default to the original order
        reordered_imgs: list[Path] = sorted(imgs, key=lambda x: extract_index(x, imgs.index(x)))
        
        # Read images as tensor from filenames
        tensor_images = []
        tensor_masks = []
        for img_path in reordered_imgs:
            if not os.path.exists(img_path):
                continue
            tensor_img = read_image(str(img_path), mode=ImageReadMode.RGB_ALPHA)  # will read as CHW
            tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0) / 255.0   # back to HWC
            tensor_images.append(tensor_img[..., :3])  # RGB
            mask = (1 - tensor_img[..., -1]).squeeze()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)    # add batch dim
            tensor_masks.append(mask)

        return torch.cat(tensor_images, dim=0), torch.cat(tensor_masks, dim=0)


class LegacyNoiseSequenceLoader(StableRenderingNode):
    
    Category = "legacy_loader"
    
    def __call__(self, 
                 data_paths: List[PATH(accept_folder=False, 
                                 accept_multiple=True, 
                                 to_folder='temp',
                                 accept_types=[".jpeg", ".png", ".bmp", ".jpg", '.npy'])],    # type: ignore
                 ) -> LATENT:
        '''load sequence of noise from img or directly from npy.'''
        
        def extract_index(data_path, i):
            filename = os.path.basename(data_path)
            if filename.split('.')[0].split('_')[-1].isdigit():  # e.g. ..._0.npy
                return int(filename.split('.')[0].split('_')[-1])
            elif filename.split('_')[0].isdigit():   # e.g. 0_....npy
                return int(filename.split('_')[0])
            return i    # default to the original order
        reordered_data_paths = sorted(data_paths, key=lambda x: extract_index(x, data_paths.index(x)))

        tensors = []
        for data_path in reordered_data_paths:
            if not os.path.exists(data_path):
                continue
            if data_path.endswith('.npy'):
                tensor = torch.from_numpy(np.load(data_path)).squeeze()
                if not len(tensor.shape) == 3:
                    raise ValueError(f"Invalid shape of noise tensor: {tensor.shape}.")
                if not (tensor.shape[-1] == 4 or tensor.shape[1] == 4): # not chw or hwc
                    raise ValueError(f"Invalid noise tensor shape: {tensor.shape}.")
                if tensor.shape[-1] == 4:
                    tensor = rearrange(tensor, 'h w c -> c h w')
            else:
                tensor = read_image(str(data_path), mode=ImageReadMode.RGB_ALPHA) / 255.0
            tensors.append(tensor)
        for t in tensors:
            if t.shape != tensors[0].shape:
                raise ValueError(f"Tensor data has inconsistent shapes: {t.shape} and {tensors[0].shape}.")

        t = torch.cat(tensors, dim=0)
        return LATENT(samples=torch.zeros_like(t), noise=t)


class LegacyIDSequenceLoader(StableRenderingNode):

    Category = "legacy_loader"

    def __call__(self,
                data_paths: List[PATH(accept_folder=False, 
                                 accept_multiple=True, 
                                 to_folder='temp',
                                 accept_types=[".jpeg", ".png", ".bmp", ".jpg", '.npy'])],  # type: ignore
                ) -> IDMap:
        def extract_index(data_path, i):
            filename = os.path.basename(data_path)
            if filename.split('.')[0].split('_')[-1].isdigit():  # e.g. ..._0.npy
                return int(filename.split('.')[0].split('_')[-1])
            elif filename.split('_')[0].isdigit():   # e.g. 0_....npy
                return int(filename.split('_')[0])
            return i    # default to the original order
        reordered_data_paths = sorted(data_paths, key=lambda x: extract_index(x, data_paths.index(x)))

        frame_indices = list(map(extract_index, reordered_data_paths))

        id_tensors = [] 
        for data_path in reordered_data_paths:
            if not os.path.exists(data_path):
                continue
            if data_path.endswith('.npy'):
                id_tensor = torch.from_numpy(np.load(data_path)).squeeze()
                if not len(id_tensor.shape) == 3:
                    raise ValueError(f"Invalid shape of id tensor: {id_tensor.shape}.")
                if not (id_tensor.shape[-1] == 4 or id_tensor.shape[1] == 4): # not chw or hwc
                    raise ValueError(f"Invalid id tensor shape: {id_tensor.shape}.")
                if id_tensor.shape[-1] == 4:
                    id_tensor = rearrange(id_tensor, 'h w c -> c h w')
            else:
                id_tensor = read_image(str(data_path), mode=ImageReadMode.RGB_ALPHA) / 255.0
            id_tensors.append(id_tensor)

        for t in id_tensors:
            if t.shape != id_tensors[0].shape:
                raise ValueError(f"Tensor data has inconsistent shapes: {t.shape} and {id_tensors[0].shape}.")
        t = torch.cat(id_tensors, dim=0)

        return IDMap(frame_indices=frame_indices, tensor=t)

    
__all__ = ['LegacyImageSequenceLoader', 'LegacyNoiseSequenceLoader', "LegacyIDSequenceLoader"]
