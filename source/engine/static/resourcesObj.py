from utils.global_utils import GetOrCreateGlobalValue, GetGlobalValue
from uuid import uuid4
from typing import Dict, TYPE_CHECKING, TypeVar, Optional, Type, Union
from pathlib import Path

import os
if TYPE_CHECKING:
    from engine.engine import Engine

def _random_str():
    return str(uuid4()).replace('-', '')

_RESOURCES_CLSES = GetOrCreateGlobalValue('_RESOURCES_CLSES', dict)
'''cls_name: cls'''

class ResourcesObjMeta(type):

    _Format_Clses:dict  # format: sub cls

    def __new__(cls, *args, **kwargs):
        cls_name = args[0]
        if cls_name in _RESOURCES_CLSES:
            return _RESOURCES_CLSES[cls_name]
        cls = super().__new__(cls, *args, **kwargs)
        
        if cls_name == 'ResourcesObj':
            return cls
        
        if not hasattr(cls, '_Format_Clses'):
            cls._Format_Clses = {}
        else:
            if cls.Format() is not None and cls.Format() not in cls._Format_Clses:
                cls._Format_Clses[cls.Format()] = cls
                
        _RESOURCES_CLSES[cls_name] = cls
        return cls

_AllBaseClsInstances: Dict[str, Dict[str, 'ResourcesObj']] = GetOrCreateGlobalValue('_AllBaseClsInstances', dict) # base_cls_name: cls_instances
'''
{ base_cls_name: {name: instance, ...} }
Dict for all instances of base classes, e.g. Mesh, Material, Texture...
'''

RC = TypeVar('RC', bound='ResourcesObj')
'''Type hint for sub-classes of ResourcesObj'''

class ResourcesObj(metaclass=ResourcesObjMeta):
    '''Class for resources that need to load to GPU before get into the main loop'''

    # region cls properties
    _Format: Optional[str] = None # format of the data, e.g. "obj" for "Mesh_OBJ"
    '''override this property to specify the format of the mesh data, e.g. "obj"'''
    _LoadOrder: int = 0
    '''override this property to specify the order of loading to GPU by ResourcesManager, the smaller the earlier'''
    _BaseName: Optional[str] = None
    '''Override this to set the base name of the class, e.g. "Mesh" for "Mesh_OBJ"'''

    _internal_id: str
    '''For internal use only, do not override this'''
    _name: str
    '''name of this obj. Each obj should have a unique name for identification.'''
    
    @classmethod
    def ClsName(cls):
        return cls.__qualname__

    @classmethod
    def BaseClsName(cls):
        baseClsName = cls._BaseName
        if baseClsName is None:
            baseClsName = cls.ClsName()
            if '_' in baseClsName:
                baseClsName = baseClsName.split('_')[0]
        return baseClsName

    @classmethod
    def FindFormat(cls:Type[RC], format)->Type[RC]:
        '''Find the sub-class that supports the given format, e.g. Mesh.FindFormat("obj") will return Mesh_OBJ.'''
        return cls._Format_Clses.get(format, None)

    @classmethod
    def Format(cls)->Optional[str]:
        if cls._Format is None:
            return None
        format = cls._Format.lower().strip()
        if format.startswith('.'):
            format = format[1:]
        return format

    @classmethod
    def BaseInstances(cls)->Dict[str, 'ResourcesObj']:
        '''
        For internal use only, do not override this
        This contains all the instances of the sub-baseclass, e.g. Mesh._ClassInstances contains all instances of its sub class.
        '''
        cls_basename = cls.BaseClsName()
        if cls_basename not in _AllBaseClsInstances:
            _AllBaseClsInstances[cls_basename] = {}
        return _AllBaseClsInstances[cls_basename]

    @classmethod
    def SupportedFormats(cls):
        return cls._Format_Clses.keys()

    def __class_getitem__(cls, item):
        return cls.FindFormat(item)

    @classmethod
    def Find(cls, name):
        return cls.BaseInstances().get(name, None)

    @classmethod
    def AllInstances(cls):
        '''All instances of this sub-baseclass, e.g. Mesh.AllInstances() contains all instances of its sub class, including MeshObj, etc.'''
        return cls.BaseInstances().values()

    @classmethod
    def _GetPathAndName(cls, path: Union[str, Path], name=None):
        '''
        Get proper path & name for cls.Load().
        If name is not given, it will be the file name without extension.
        In that case, if there is already a resource with the same name, a number will be added to the end of the name to avoid name conflict.
        '''
        if isinstance(path, Path):
            path = str(path)
        if name is None:
            name = os.path.basename(path).split('.')[0]
            count = 0
            newName = name
            while cls.Find(newName) is not None:
                count += 1
                newName = f'{name}_{count}' # try to avoid name conflict
            name = newName
        elif cls.Find(name) is not None:
            raise ValueError(f'Resources type {cls.BaseClsName()} with name "{name}" already exists')
        return path, name
    # endregion

    def __new__(cls: Type[RC], name, *args, **kwargs)->RC:
        if name in cls.BaseInstances():
            return cls.BaseInstances()[name]    # type: ignore
        else:
            obj = super().__new__(cls)  # type: ignore
            obj._name = name
            obj._internal_id = _random_str()
            cls.BaseInstances()[name] = obj
            return obj

    def __init__(self, name: str, *args, **kwargs):
        pass

    @property
    def name(self):
        return self._name

    _engine = None
    @property
    def engine(self)->'Engine':
        if self.__class__._engine is None:
            engine = GetGlobalValue('_ENGINE_SINGLETON')
            if engine is None:
                raise ValueError('No engine instance found. Engine must be initialized before any resources.')
            self.__class__._engine = engine
        return self.__class__._engine

    # region abstract methods

    def sendToGPU(self):
        '''Override this to send the data to GPU'''
        pass

    def clear(self):
        '''Override this to clear the data from GPU'''
        pass

    # endregion



__all__ = ['ResourcesObj', 'ResourcesObjMeta']