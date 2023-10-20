from .utils.sortableElement import SortableElement
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple


class CorrespondenceMap:
    r"""
    CorrespondenceMap instances should have the following structure:
    {
        'vertex_id_1': [([pixel_xpos_1, pixel_ypos_1], frame_number), ([pixel_xpos_2, pixel_ypos_2], frame_number), ...],
        'vertex_id_2': [([pixel_xpos_1, pixel_ypos_1], frame_number), ([pixel_xpos_2, pixel_ypos_2], frame_number), ...],
        ...
    }
    where vertex_ids are unique
    """

    def __init__(self,
                 correspondence_map: dict, width: int = None, height: int = None, num_frames: int = None):
        # avoid calling directly, should be initiated using classmethods
        self._correspondence_map = correspondence_map
        self._width = width
        self._height = height
        self._num_frames = num_frames

    def __str__(self):
        return self._correspondence_map.__str__()

    @property
    def Map(self) -> dict:
        return self._correspondence_map

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> tuple:
        return (self._width, self._height)

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @classmethod
    def from_existing_directory_img(cls,
                                    directory: str,
                                    num_frames: int = None,
                                    pixel_position_callback: Callable[[int, int], Tuple[int, int]] = None,
                                    enable_strict_checking=True
                                    ):
        r"""
        Create CorrespondenceMap instance from using the images in an existing directory.
        Directory should exist and numeric values should be present in the filename for every image files.
        The numeric values will be used as the key to sort id maps into ascending order for the construction of CorrespondenceMap
        Files not ending with ('.jpeg', '.png', '.bmp', '.jpg') are considered as non-image files, which will be skipped.

        :param directory: directory where id maps are stored as images
        :param num_frames: first n frames to be used for building correspondence map, all frames will be used if not specified
        :param pixel_position_callback: callback function to be applied on pixel position read from frames
        :param enable_strict_checking: when enabled, check uniqueness, only one pixel position should be added to the same id in every frame,
                                        when disabled, only the first pixel position will be added to the same id in every frame, subsequent pixels will be ignored

        """
        # Load and sort image maps from directory according to frame number
        assert os.path.exists(directory), f"Directory {directory} not found"
        id_data_container = []
        for i, file in enumerate(os.listdir(directory)):
            if not file.endswith((".jpeg", ".png", ".bmp", ".jpg")):
                print(f"[INFO] Skipping non-image file {file}")
                continue
            match = re.search(r"\d+", file)
            if match:
                frame_idx = int(match.group())
                id_data_img = Image.open(os.path.join(directory, file))
                if i == 0:
                    width, height = id_data_img.size
                id_data_array = np.array(id_data_img)
                id_data_container.append(SortableElement(value=frame_idx, object=id_data_array))
            else:
                raise RuntimeError(f"{file} has no numeric component in filename.")

        sorted_ids = sorted(id_data_container)
        if num_frames is not None:
            sorted_ids = sorted_ids[:num_frames]
        else:
            num_frames = len(sorted_ids)
        # Prepare correspondence map
        print("[INFO] Preparing correspondence map...")
        corr_map = {}
        for frame_idx, id_data in tqdm(enumerate(sorted_ids), total=len(sorted_ids)):
            assert len(id_data.Object.shape) >= 2, "id_data should be at least 2D."
            # iterate elements, where elements have shape with first 2 dimensions dropped
            # add pixel position [i, j] to corr_map dictionary with tuple(id_key) as key
            for i, row in enumerate(id_data.Object):
                for j, id in enumerate(row):
                    if np.array_equal(id, np.zeros_like(id)):
                        continue
                    id_key = tuple(id)
                    pix_xpos, pix_ypos = pixel_position_callback(i, j) if pixel_position_callback is not None else (i, j)
                    if corr_map.get(id_key) is None:
                        corr_map[id_key] = [([i, j], frame_idx)]
                    else:
                        # check uniqueness, only one pixel position should be added to the same id per every frame
                        if enable_strict_checking:
                            assert len(corr_map[id_key]) < frame_idx, f"Corr_map[key={id_key}] value={corr_map[id_key]} has already appended a pixel position at frame index {frame_idx}."
                        else:
                            if len(corr_map[id_key]) >= frame_idx:
                                continue
                        corr_map[id_key].append(([i, j], frame_idx))
        return CorrespondenceMap(corr_map, width, height, num_frames)


__all__ = ['CorrespondenceMap']
