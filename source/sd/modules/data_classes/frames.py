from utils.data_struct import SortableElement
from typing import Any
import os

# TODO: Make DataClass as the parent for all other dataclasses 
class Frames:
    def __init__(self, data: Any):
        self._data = data
    
    @property
    def Data(self)->list:
        return self._data
    
    def __str__(self):
        return self._data.__str__()
    
    @classmethod
    def from_existing_directory(cls, directory: str):
        pass