import numpy
from .utils.sortableElement import SortableElement
from .. import log_utils as logu, config
from PIL import Image
from utils.path_utils import MAP_OUTPUT_DIR
import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple

CACHE_DIR = "./.cache"
# MAP_OUTPUT_DIR = config.test_dir / 'boat'


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
    def Load_ID_Data_From_Dir(cls,
                              directory: str = None,
                              num_frames: int = None,
                              pixel_position_callback: Callable[[int, int], Tuple[int, int]] = None,
                              enable_strict_checking: bool = True,
                              use_cache: bool = False):
        r"""
        Create CorrespondenceMap instance from using the images in an existing output path .
        Directory should exist and numeric values should be present in the filename for every image files.
        The numeric values will be used as the key to sort id maps into ascending order for the construction of CorrespondenceMap
        Files not ending with ('.jpeg', '.png', '.bmp', '.jpg') are considered as non-image files, which will be skipped.

        :param directory: directory where id maps are stored as images. If not given, the default output path with lastest timestamp will be used.
        :param num_frames: first n frames to be used for building correspondence map, all frames will be used if not specified
        :param pixel_position_callback: callback function to be applied on pixel position read from frames
        :param enable_strict_checking: when enabled, check uniqueness, only one pixel position should be added to the same id in every frame,
                                        when disabled, only the first pixel position will be added to the same id in every frame, subsequent pixels will be ignored
        :param use_cache: whether to use cache to speed up loading. Cache will be generated if not found.

        """
        if directory is None:
            current_dirs = os.listdir(MAP_OUTPUT_DIR)
            if len(current_dirs) == 0:
                raise FileNotFoundError(f"No output directory found in {MAP_OUTPUT_DIR}")
            directory = os.path.join(MAP_OUTPUT_DIR, sorted(current_dirs)[-1], 'id')

        if use_cache:
            cache_corr_map = cls._load_correspodence_map_from_cache(directory)
            if cache_corr_map is not None:
                return cache_corr_map

        # Load and sort image maps from directory according to frame number
        assert os.path.exists(directory), f"Directory {directory} not found"
        id_data_container = []
        for i, file in enumerate(os.listdir(directory)):
            if not file.endswith((".jpeg", ".png", ".bmp", ".jpg", ".npy")):
                logu.warn(f"Skipping non-image file {file}")
                continue
            match = re.search(r"\d+", file)
            if match:
                frame_idx = int(match.group())
                id_data_array = numpy.load(os.path.join(directory, file), allow_pickle=True)  # [height, width, ...]
                if i == 0:
                    width, height = id_data_array.shape[1], id_data_array.shape[0]
                id_data_container.append(SortableElement(value=frame_idx, object=id_data_array))
            else:
                raise RuntimeError(f"{file} has no numeric component in filename.")

        sorted_ids = sorted(id_data_container)
        if num_frames is not None:
            sorted_ids = sorted_ids[:num_frames]
        else:
            num_frames = len(sorted_ids)
        # Prepare correspondence map
        logu.info("Preparing correspondence map...")
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
                        corr_map[id_key] = [([pix_xpos, pix_ypos], frame_idx)]
                    else:
                        # check uniqueness, only one pixel position should be added to the same id per every frame
                        if enable_strict_checking:
                            assert len(corr_map[id_key]) < frame_idx, f"Corr_map[key={id_key}] value={corr_map[id_key]} has already appended a pixel position at frame index {frame_idx}."
                        else:
                            if len(corr_map[id_key]) >= frame_idx:
                                continue
                        corr_map[id_key].append(([pix_xpos, pix_ypos], frame_idx))

        ret = CorrespondenceMap(corr_map, width, height, num_frames)
        if use_cache:
            cls._save_correspondence_map_to_cache(directory, ret)
        return ret

# TODO: integrate utils.make_corr_map
    @staticmethod
    def _get_cache_path(img_from_dir: str):
        return os.path.join(CACHE_DIR, os.path.basename(os.path.dirname(img_from_dir)), 'corr_map.pkl')

    @staticmethod
    def _load_correspodence_map_from_cache(img_from_dir: str):
        cached_fname = CorrespondenceMap._get_cache_path(img_from_dir)
        if os.path.exists(cached_fname):
            try:
                with open(cached_fname, 'rb') as f:
                    corr_map: CorrespondenceMap = pickle.load(f)
                logu.success(f"[SUCCESS] Correspondence map loaded from {cached_fname}")
                return corr_map
            except Exception as e:
                os.remove(cached_fname)
        return None

    @staticmethod
    def _save_correspondence_map_to_cache(img_from_dir: str, corr_map):
        cache_fname = CorrespondenceMap._get_cache_path(img_from_dir)
        if not os.path.exists(cache_fname):
            os.makedirs(os.path.dirname(cache_fname), exist_ok=True)
            with open(cache_fname, 'wb') as f:
                pickle.dump(corr_map, f)
            logu.success(f"[SUCCESS] Correspondence map created and cached to {cache_fname}")


__all__ = ['CorrespondenceMap']
