import json
from attr import attrib, attrs
from typing import List, Union
from pathlib import Path

from .resources_obj import ResourcesObj

# TODO

@attrs(eq=False, repr=False)
class Scene(ResourcesObj): 
    '''Scene object, containing all objects's initial states & information.'''

    BaseClsName = 'Scene'

    config: dict = attrib(factory=dict, kw_only=True)
    '''engine configuration for the scene, such as camera, light, etc.'''
    objects: List[dict] = attrib(factory=list, kw_only=True)
    '''list of objects in the scene, e.g. GameObjects, Components, etc.'''
    _immediate_load: bool = attrib(default=False, kw_only=True)
    '''scene object will not load immediately by default'''

    @classmethod
    def Load(cls, path: Union[str, Path]):
        with open(path, 'r') as f:
            data = json.load(f)
        config = data.get('config', {})
        objects = data.get('objects', [])
        return cls(config=config, objects=objects)
    
    def load(self):
        '''TODO'''


__all__ = ['Scene']