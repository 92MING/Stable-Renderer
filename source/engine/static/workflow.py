from typing import Union
from pathlib import Path

class Workflow:
    '''
    The workflow for ComfyUI.
    Each workflow represents a rendering process/pipeline.
    '''
    @classmethod
    def Load(cls, path: Union[str, Path]):
        '''
        Load a workflow from a file.
        '''
        pass
    
    
    
__all__ = ['Workflow']