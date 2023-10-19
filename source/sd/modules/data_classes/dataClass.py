from .utils.sortableElement import SortableElement
from typing import Any
import os

# TODO: Make DataClass as the parent for all other dataclasses 
class DataClass:
    def __init__(self, data: Any):
        self._data = data
    
    @property
    def Data(self):
        return self._data
    
    @classmethod
    def from_existing_directory(cls,
                                directory: str,):
        assert os.path.exists(directory), f"Directroy {directory} not found"
        pass
