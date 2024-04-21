import os
import re

from typing import List
from PIL import Image
from PIL.Image import Image as PILImage

from common_utils.data_struct import SortableElement

# ! Deprecated
class ImageFrames:
    def __init__(self, data: List[PILImage]):
        self._data = data

    @property
    def Data(self) -> List[PILImage]:
        return self._data 

    @classmethod
    def from_existing_directory(cls,
                                directory: str,
                                frame_start: int,
                                num_frames: int):
        assert os.path.exists(directory), f"Directory {directory} not found"
        assert frame_start >= 0, f"frame_start must be non-negative"
        assert num_frames > 0, f"num_frames must be positive"

        file_filter = lambda fname: fname.endswith((".jpeg", ".png", ".bmp", ".jpg"))
        extract_key = lambda fname: int(re.search(r"\d+", fname).group())

        data_container = []
        for filename in os.listdir(directory):
            if file_filter(filename):
                key = extract_key(filename)
                with Image.open(os.path.join(directory, filename)) as image:
                    data_container.append(SortableElement(value=key, object=image))
    
        data_container = [
            d.Object for d in sorted(data_container)[frame_start: frame_start+num_frames]
        ]
        return ImageFrames(data_container)

__all__ = ['ImageFrames']