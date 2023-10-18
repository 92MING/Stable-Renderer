from .manager import Manager
from typing import List
import os

class OutputManager(Manager):
    def __init__(self, output_dir: str,
                 output_subfolders: List[str],
                 save_map_per_num_frame: int):
        super().__init__()
        self._output_dir = output_dir
        self._output_subfolders = output_subfolders
        assert save_map_per_num_frame > 0
        self._save_map_per_num_frame = save_map_per_num_frame

    @property
    def OutputDir(self):
        return self._output_dir

    @property
    def SaveMapPerNumFrame(self):
        return self._save_map_per_num_frame

    def _onFrameBegin(self):
        os.makedirs(self._output_dir, exist_ok=True)
        if self._output_subfolders is not None:
            for subfolder_dir in self._output_subfolders:
                cur_subfolder_path = os.path.join(self._output_dir, subfolder_dir)
                os.makedirs(cur_subfolder_path, exist_ok=True)

__all__ = ['OutputManager']