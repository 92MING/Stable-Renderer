import os
import re
import torch

from pathlib import Path
from torchvision.io import read_image, ImageReadMode

from comfyUI.types import *
from stable_renderer.src.data_classes import CorrespondenceMap, ImageFrames


class CorrespondenceMapLoader(StableRendererNodeBase):

    Category = "loader"

    def __call__(self,
                 directory: Path,
                 num_frames: INT(min=0),  # type: ignore
        ) -> CorrespondenceMap:
        return CorrespondenceMap.from_existing(directory, num_frames)


class ImageSequenceLoader(StableRendererNodeBase):
    
    Category = "loader"
    
    def __call__(self, 
                 directory: Path,
                 frame_start: INT(min=0) = 0,
                 num_frames: INT(min=1) = 16
    ) -> IMAGE:
        """Load image sequence from a given folder

        Note: The filename of images should contain a number, indicating its frame index in
        a sequence. E.g. depth_0.png, depth_2.png. Undefined behaviour will occur if such indicator
        is not found.

        Args:
            directory (str): The absolute path to the desired folder
            frame_start (int, optional): The first returning frame. Defaults to 0.
            num_frames (int, optional): Number of frames to be returned. Defaults to 16.

        Returns:
            torch.Tensor: Tensor of shape (num_frames, height, width, channels)
        """
        assert os.path.exists(directory), f"Directory {directory} not found"
        assert frame_start >= 0, "frame_start must be no-negative"
        assert num_frames > 0, "num_frames must be positive"

        file_filter = lambda fname: fname.endswith((".jpeg", ".png", ".bmp", ".jpg"))
        get_frame_number = lambda fname: int(re.search(r"\d+", fname).group())

        # Sort filenames by the index extracted from filename
        filenames = list(filter(file_filter, os.listdir(directory)))
        filenames.sort(key=get_frame_number)

        # Read images as tensor from filenames
        tensor_images = []
        for filename in filenames[frame_start: frame_start+num_frames]:
            tensor_img = read_image(os.path.join(directory, filename), mode=ImageReadMode.RGB)
            tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0) / 255.0
            tensor_images.append(tensor_img)

        return torch.cat(tensor_images, dim=0) 