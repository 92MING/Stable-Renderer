from map import Map
from utils.sortableElement import SortableElement
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm


class CorrespondenceMap(Map):
    r"""
    CorrespondenceMap instances should have the following structure:
    {
        'vertex_id_1': [[pixel_xpos_1, pixel_ypos_1], [pixel_xpos_2, pixel_ypos_2], ...],
        'vertex_id_2': [[pixel_xpos_1, pixel_ypos_1], [pixel_xpos_2, pixel_ypos_2], ...],
        ...
    }
    where vertex_ids are unique
    """
    def __init__(self,
                 correspondence_map: dict):
        # avoid calling directly, should be initiated using classmethods
        self._correspondence_map = correspondence_map

    @property
    def CorrespondenceMap(self)->dict:
        return self._correspondence_map

    @classmethod
    def from_existing_directory_img(cls, directory: str, enable_strict_checking=True):
        r"""
        Create CorrespondenceMap instance from using the images in an existing directory.
        Directory should exist and numeric values should be present in the filename for every image files.
        The numeric values will be used as the key to sort id maps into ascending order for the construction of CorrespondenceMap
        Files not ending with ('.jpeg', '.png', '.bmp', '.jpg') are considered as non-image files, which will be skipped.
 
        :param directory: directory where id maps are stored as images
        :param enable_strict_checking: when enabled, check uniqueness, only one pixel position should be added to the same id in every frame,
                                        when disabled, only the first pixel position will be added to the same id in every frame, subsequent pixels will be ignored

        """
        # Load and image maps from directory according to frame number
        assert os.path.exists(directory), f"Directory {directory} not found"
        id_data_container = []
        for file in os.listdir(directory):
            if not file.endswith((".jpeg", ".png", ".bmp", ".jpg")):
                print(f"[INFO] Skipping non-image file {file}")
                continue
            match = re.search(r"\d+", file)
            if match:
                frame_number = int(match.group())
                id_data_img = Image.open(os.path.join(directory, file))
                id_data_array = np.array(id_data_img)
                id_data_container.append(SortableElement(value=frame_number, object=id_data_array))
            else:
                raise RuntimeError(f"{file} has no numeric component in filename.")
        sorted_ids = sorted(id_data_container)
        
        # Prepare correspondence map
        print("[INFO] Preparing correspondence map...")
        corr_map = {}
        for frame_number, id_data in tqdm(enumerate(sorted_ids), total=len(sorted_ids)):
            assert len(id_data.Object.shape) >= 2, "id_data should be at least 2D."
            # iterate elements, where elements have shape with first 2 dimensions dropped
            # add pixel position [i, j] to corr_map dictionary with tuple(id_key) as key
            for i, row in enumerate(id_data.Object):
                for j, id in enumerate(row):
                    if np.array_equal(id, np.zeros_like(id)):
                        continue
                    id_key = tuple(id)
                    if corr_map.get(id_key) is None:
                        corr_map[id_key] = [[i, j]]
                    else:
                        # check uniqueness, only one pixel position should be added to the same id per every frame
                        if enable_strict_checking:
                            assert len(corr_map[id_key]) < (frame_number + 1), f"Corr_map[key={id_key}] value={corr_map[id_key]} has already appended a pixel position at frame number {frame_number+1}."
                        else:
                            if len(corr_map[id_key]) >= (frame_number + 1):
                                continue
                        corr_map[id_key].append([i, j])
        return CorrespondenceMap(corr_map)


__all__ = ['CorrespondenceMap']
if __name__ == '__main__':
    map = CorrespondenceMap.from_existing_directory_img(
        '/Users/cyruss081115/Documents/Study/Uni/research/stable_render/Stable-Renderer/rendered_frames/maps_per_30_frames/id',
        enable_strict_checking=True)

    print(len(map.CorrespondenceMap))