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

class SimpleVideoCombine(StableRendererNodeBase):
     
    Category = "video"
    
    def __call__(
        self,
        images: IMAGE,
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
        
        # get output information
        output_dir = (folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory())
        
        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])
        
        # comfy counter workaround
        max_counter = 0

        # Loop through the existing files
        matcher = re.compile(fr"{re.escape(filename)}_(\d+)\D*\..+")
        for existing_file in os.listdir(full_output_folder):
            # Check if the file matches the expected format
            match = matcher.fullmatch(existing_file)
            if match:
                # Extract the numeric portion of the filename
                file_counter = int(match.group(1))
                # Update the maximum counter value if necessary
                if file_counter > max_counter:
                    max_counter = file_counter

        # Increment the counter by 1 to get the next available value
        counter = max_counter + 1
        
        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(_tensor_to_bytes(images[0])).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        image_kwargs = {}
        image_kwargs['disposal'] = 2

        file = f"{filename}_{counter:05}.gif"
        file_path = os.path.join(full_output_folder, file)
        images_bytes = _tensor_to_bytes(images)
        if pingpong:
            images_bytes = np.concatenate((images_bytes, images_bytes[-2:0:-1]))  # type: ignore
        frames = [Image.fromarray(f) for f in images_bytes]
        # Use pillow directly to save an animated image
        frames[0].save(
            file_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / frame_rate),
            loop=loop_count,
            compress_level=4,
            **image_kwargs
        )
        output_files.append(file_path)

        return UIImage(images, 
                       filename=file, 
                       subfolder=subfolder, 
                       type="output" if save_output else "temp",
                       animated=True)
        
__all__ = ['SimpleVideoCombine']