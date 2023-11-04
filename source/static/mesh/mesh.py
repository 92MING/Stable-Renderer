import os
from static.resourcesObj import ResourcesObj
from utils.global_utils import GetOrAddGlobalValue, SetGlobalValue

class Mesh(ResourcesObj):
    _BaseName = 'Mesh'

    def __init__(self, name):
        super().__init__(name)
        currentID = GetOrAddGlobalValue('_MeshCount', 1) # 0 is reserved
        self._meshID = currentID # for corresponding map
        SetGlobalValue('_MeshCount', currentID+1)

    @property
    def meshID(self):
        return self._meshID

    def load(self, path:str):
        '''Load data from file. Override this function to implement loading data from file'''
        raise NotImplementedError
    def draw(self, slot:int=None):
        '''Draw the mesh'''
        raise NotImplementedError

    @classmethod
    def Load(cls, path: str, name=None)->'Mesh':
        '''
        load a mesh from file
        :param path: path of the mesh file
        :param name: name is used to identify the mesh. If not specified, the name will be the path of the mesh file
        :return: Mesh object
        '''
        path, name = cls._GetPathAndName(path, name)
        if '.' not in os.path.basename(path):
            raise ValueError('path must contain file extension')
        ext = path.split('.')[-1]
        formatCls = cls.FindFormat(ext)
        if formatCls is None:
            raise ValueError(f'unsupported mesh format: {ext}')
        mesh = formatCls(name)
        mesh.load(path)
        return mesh

__all__ = ['Mesh']