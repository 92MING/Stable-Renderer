import numpy
import os
import re
import pickle
import random
import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Callable, Tuple, Union, List, TypeVar, Type, Optional

from .common import Rectangle
from common_utils.debug_utils import DefaultLogger
from common_utils.os_utils import is_windows, is_mac
from common_utils.data_struct import SortableElement
from common_utils.path_utils import MAP_OUTPUT_DIR

T = TypeVar('T', bound='CorrespondenceMap')


# TODO: Same number of dropout over the time steps
# TODO: Use the same noise map?
# TODO: Varying noise map

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
                 correspondence_map: dict,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 num_frames: Optional[int] = None):
        # avoid calling directly, should be initiated using classmethods
        self._correspondence_map = correspondence_map
        self._width = width
        self._height = height
        self._num_frames = num_frames

    def __str__(self):
        return self._correspondence_map.__str__()
    
    def __len__(self):
        return self._correspondence_map.__len__()

    @property
    def Map(self) -> dict:
        return self._correspondence_map

    @property
    def width(self) -> int:
        return self._width  # type: ignore

    @property
    def height(self) -> int:
        return self._height # type: ignore

    @property
    def size(self) -> tuple:
        return (self._width, self._height)

    @property
    def num_frames(self) -> int:
        return self._num_frames # type: ignore

    @classmethod
    def from_existing(cls,
                     directory: Union[str, Path, None] = None,
                     num_frames: Optional[int] = None,
                     get_pixel_position_callback: Optional[Callable[[int, int], Tuple[int, int]]] = None,
                     enable_strict_checking: bool = True,):
        r"""
        Create CorrespondenceMap instance from using the numpy files in an existing output path .
        Directory should exist and numeric values should be present in the filename for every files.
        The numeric values will be used as the key to sort id maps into ascending order for the construction of CorrespondenceMap
        Files not ending with .npy are considered as non-numpy files, which will be skipped.

        :param directory: This can be:
            * directory where id maps are stored as .npy. If not given, the default output path with lastest timestamp will be used.
            * the corresponding cache path
        :param num_frames: first n frames to be used for building correspondence map, all frames will be used if not specified
        :param get_pixel_position_callback: callback function to be applied on pixel position read from frames
        :param enable_strict_checking: when enabled, check uniqueness, only one pixel position should be added to the same id in every frame,
                                        when disabled, only the first pixel position will be added to the same id in every frame, subsequent pixels will be ignored

        """
        if directory is None:
            current_dirs = os.listdir(MAP_OUTPUT_DIR)
            if len(current_dirs) == 0:
                raise FileNotFoundError(f"No output directory found in {MAP_OUTPUT_DIR}")
            directory = os.path.abspath(os.path.join(MAP_OUTPUT_DIR, sorted(current_dirs)[-1], 'id'))

        if is_windows():
            directory = directory.replace('/', '\\')

        cache_path = None
        if os.path.isfile(directory) and directory.endswith('.pkl'):
            cache_path = directory
        elif 'corr_map.pkl' in os.listdir(directory):   # if the cache file is in the directory
            cache_path = os.path.join(directory, 'corr_map.pkl')
        elif directory.endswith('id') and 'corr_map.pkl' in os.listdir(os.path.join(directory, '..')):
            cache_path = os.path.join(directory, '..', 'corr_map.pkl')
        if cache_path is not None:
            cache_corr_map = cls.load_correspondence_map_from_cache(cache_path)
            if cache_corr_map is not None:
                return cache_corr_map
        
        assert os.path.exists(directory), f"Directory {directory} not found"
        assert os.path.isdir(directory), f"{directory} is not a directory"
        
        if not directory.endswith('id') and 'id' in os.listdir(directory):  # switch to id directory automatically
            directory = os.path.join(directory, 'id')
        
        # Load and sort image maps from directory according to frame number
        id_data_container = []
        for i, file in enumerate(os.listdir(directory)):
            if not file.endswith(".npy"):
                DefaultLogger.warn(f"[WARNING] Skipping non-numpy file {file}")
                continue
            match = re.search(r"\d+", file)
            if match:
                frame_idx = int(match.group())
                id_data_array = numpy.load(os.path.join(directory, file), allow_pickle=True)
                if i == 0:
                    width, height = id_data_array.shape[1], id_data_array.shape[0]
                id_data_container.append(SortableElement(value=frame_idx, object=id_data_array))    # type: ignore
            else:
                raise RuntimeError(f"{file} has no numeric component in filename.")

        sorted_ids = sorted(id_data_container)
        if num_frames is not None:
            sorted_ids = sorted_ids[:num_frames]
        else:
            num_frames = len(sorted_ids)
            
        # prepare correspondence map
        DefaultLogger.info("Preparing correspondence map...")
        corr_map = {}
        for frame_idx, id_data in tqdm(enumerate(sorted_ids), total=len(sorted_ids)):
            assert len(id_data.Object.shape) >= 2, "id_data should be at least 2D."
            # iterate elements, where elements have shape with first 2 dimensions dropped
            for i, row in enumerate(id_data.Object):
                for j, id in enumerate(row):
                    if np.array_equal(id, np.zeros_like(id)):
                        continue    # means no id
                    
                    id_key = tuple(id)
                    pix_xpos, pix_ypos = get_pixel_position_callback(i, j) if get_pixel_position_callback is not None else (i, j)
                    
                    if corr_map.get(id_key) is None:
                        corr_map[id_key] = [([pix_xpos, pix_ypos], frame_idx)]
                    else:
                        # check uniqueness, only one pixel position should be added to the same id per every frame
                        if enable_strict_checking:
                            assert len(corr_map[id_key]) < frame_idx, f"Corr_map[key={id_key}] value={corr_map[id_key]} has already appended a pixel position at frame index {frame_idx}."
                        # else:
                        #     if len(corr_map[id_key]) >= frame_idx:
                        #         continue
                        corr_map[id_key].append(([pix_xpos, pix_ypos], frame_idx))

        ret = CorrespondenceMap(corr_map, width, height, num_frames)
        cache_path = os.path.join(directory, ('..' if directory.endswith('id') else ''), 'corr_map.pkl')
        ret.save_cache(path=cache_path)
        return ret

# TODO: integrate utils.make_corr_map
    @ classmethod
    def load_correspondence_map_from_cache(cls: Type[T], path: str)->T:
        '''
        Load the correspondence map from a cache file, if it exists.
        
        Args:
            * path: the full path to the cache file. If a dir is given, the file will be loaded from '{path}/corr_map.pkl'
        '''
        if os.path.exists(path) and os.path.isdir(path):
            path = os.path.join(path, 'corr_map.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Correspondence map cache file not found at {path}")
        
        with open(path, 'rb') as f:
            corr_map: T = pickle.load(f)
            DefaultLogger.success(f"Correspondence map loaded from {path}")
            return corr_map

    def save_cache(self, path: str):
        '''
        Save the correspondence map to a file, for quick loading in the future.
        
        Args:
            * path: the full path to save the cache file. If a dir is given, the file will be saved as '{path}/corr_map.pkl'
        '''
        if os.path.exists(path) and os.path.isdir(path):
            path = os.path.join(path, 'corr_map.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        DefaultLogger.success(f"Correspondence map created and cached to {path}")
    
    def dropout_index(self, probability: float, seed: int):
        """Randomly drop indices in correspondence map with `probability`.
        Note that this proess is irreversible.

        Args:
            probability (float): Probability of dropout. The value should be in [0, 1]
            seed (int): Seed for reproducing the event
        """
        assert 0 <= probability <= 1
        random.seed(seed)
        DefaultLogger.info(f"Enabled correspondence map dropout of p={probability}, seed={seed}")
        # Convert the dictionary to a list of key-value pairs
        corr_map_copy = self._correspondence_map.copy()
        for key in list(corr_map_copy.keys()):
            if random.random() < probability:
                del corr_map_copy[key]
        self._correspondence_map = corr_map_copy

    def dropout_in_rectangle(self, 
                             rectangle: Union[Rectangle, Tuple[Tuple[int, int], Tuple[int, int]]],
                             at_frame: int):
        """Dropout indices within the `rectangle` at frame index = `at_frame`

        Args:
            rectangle (Rectangle | Tuple[Tuple[int, int], Tuple[int, int]]): Bounding rectangle for the 
                points to be dropped, the coordinates of rectangle should not exceed frame size
            at_frame: frame to apply rectangle
        """
        assert 0 <= at_frame <= self.num_frames
        if isinstance(rectangle, Rectangle):
            assert self.width >= max(rectangle.top_left[0], rectangle.bottom_right[0])
            assert self.height >= max(rectangle.top_left[1], rectangle.bottom_right[1])
        elif isinstance(rectangle, tuple):
            top_left, bottom_right = rectangle
            assert self.width >= max(top_left[0], bottom_right[0])
            assert self.height >= max(top_left[1], bottom_right[1])
        else:
            raise ValueError(f"Data type of {type(rectangle)} is not supported for rectangle")

        def is_in_rectangle(track: Tuple[List[int], int]):
            """Helper function to check if certain coordinates are in a rectangle
            
            Args:
                track (Tuple[List[int], int]): Track should be in a structure of ([pixel_pos_x, pixel_pos_y], frame_number)
            Raises:
                ValueError: If the data type of rectangle is neither `Rectangle` nor `Tuple` 
            """
            position, frame_number = track
            if frame_number != at_frame:
                return False
            # Assumes top left of an image is the origin [0, 0]
            if isinstance(rectangle, Rectangle):
                return rectangle.is_in_rectangle(position)  # type: ignore
            elif isinstance(rectangle, tuple):
                top_left, bottom_right = rectangle
                return (position[0] < bottom_right[0] and position[0] > top_left[0]) and  \
                    (position[1] < bottom_right[1] and position[1] > top_left[1])
            else:
                raise ValueError
        
        corr_map_copy = self._correspondence_map.copy()
        for key in self._correspondence_map.keys():
            pixel_tracks = corr_map_copy[key]
            for t in pixel_tracks:
                if is_in_rectangle(t):
                    del corr_map_copy[key]
                    break
        self._correspondence_map = corr_map_copy
    
    def merge_nearby(self, distance: int):
        merged_corr_map = {}
        for key in self._correspondence_map.keys():
            object_id, material_id, texture_x, texture_y = key
            trace = merged_corr_map.get((object_id, material_id, texture_x // distance, texture_y // distance), [])
            for t in self._correspondence_map[key]:
                if not t in trace:
                    trace.append(t)
            merged_corr_map[(object_id, material_id, texture_x // distance, texture_y // distance)] = trace
        DefaultLogger.info(f"Correspondence map vertices before merge: {len(self._correspondence_map)}, after merge: {len(merged_corr_map)}")
        self._correspondence_map = merged_corr_map


__all__ = ['CorrespondenceMap']