'''
Nodes for video related processing.
 
Copy & modified from these modules:
    - Video Helper Suite
'''

import os
import json
import numpy as np
import re
import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

from comfyUI.types import *


def _tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))

def _tensor_to_bytes(tensor):
    return _tensor_to_int(tensor, 8).astype(np.uint8)

class SimpleVideoCombine(StableRenderingNode):
     
    Category = "video"
    
    def __call__(
        self,
        images: IMAGE,
        alpha_threshold: FLOAT(min=0, max=1, step=0.01) = 0.5,
        enable_alpha_threshold: bool = True,
        frame_rate: INT(min=1, step=1) = 8,  # type: ignore
        loop_count: INT(min=0, max=100, step=1) = 0,  # type: ignore
        filename_prefix: str = "",
        pingpong: bool = False,
        save_output: bool =True,
        prompt:PROMPT = None,  # type: ignore
        extra_pnginfo: EXTRA_PNG_INFO = None,  # type: ignore
    )->UIImage:
        '''
        Modify from "Video Helper Suite", adds some parameter especially for transparent video.
        Combine a list of images into a gif. (for complicated combination, use "Video Helper Suite" instead).
        
        Args:
            images: A list of images to be combined.
            frame_rate: The frame rate of the output gif.
            loop_count: The loop count of the output gif.
            filename_prefix: The prefix of the output gif file name.
            pingpong: Whether to add a pingpong effect to the output gif(play the gif in reverse after playing it forward)
            save_output: Whether to save the output gif.
        '''
        if enable_alpha_threshold:
            # check if has alpha channel, if yes, if alpha is less than threshold, set it to 0
            for i in range(len(images)):
                if images[i].mode == 'RGBA':
                    images[i] = images[i][..., :3] * (images[i][..., 3:] > alpha_threshold)
                else:   # add alpha channel to tensor
                    images[i] = torch.cat([images[i], torch.ones_like(images[i][:, :, :1])], dim=-1)
                    
        return UIImage(images,
                       type='temp' if not save_output else 'output',
                       frame_rate=frame_rate,
                       loop_count=loop_count,
                       prefix=filename_prefix,
                       pingpong=pingpong,
                       animated=True,
                       prompt=prompt,
                       png_info=extra_pnginfo)
        
__all__ = ['SimpleVideoCombine']