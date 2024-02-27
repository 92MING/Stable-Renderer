from .frames import Frames
from utils.data_struct import SortableElement
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Callable, List
import numpy as np
import os
import re

class ImageFrames(Frames):
    def __init__(self, data: List[PILImage]):
        super().__init__(data)
    
    @classmethod
    def from_existing_directory(cls,
                                directory: str,
                                num_frames: int):
        assert os.path.exists(directory), f"Directory {directory} not found"
        file_filter = lambda fname: fname.endswith((".jpeg", ".png", ".bmp", ".jpg"))
        extract_key = lambda fname: int(re.search(r"\d+", fname).group())
        data_container = []
        for filename in os.listdir(directory):
            if file_filter(filename):
                key = extract_key(filename)
                image = Image.open(os.path.join(directory, filename))
                data_container.append(SortableElement(value=key, object=image))
        data_container = [d.Object for d in sorted(data_container)[:num_frames]]
        return ImageFrames(data_container)

__all__ = ['ImageFrames']