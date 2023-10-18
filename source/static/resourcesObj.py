from utils.global_utils import GetOrAddGlobalValue, GetGlobalValue
import os

_RESOURCES_CLSES = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()) #cls_name: cls
class ResourcesObjMeta(type):
    _Format_Clses:dict = None # format: sub cls
    def __new__(cls, *args, **kwargs):
        cls_name = args[0]
        if cls_name in _RESOURCES_CLSES:
            return _RESOURCES_CLSES[cls_name]
        cls = super().__new__(cls, *args, **kwargs)
        if cls_name == 'ResourcesObj':
            return cls
        if cls._Format_Clses is None:
            cls._Format_Clses = {}
        else:
            if cls.Format() is not None and cls.Format() not in cls._Format_Clses:
                cls._Format_Clses[cls.Format()] = cls
        _RESOURCES_CLSES[cls_name] = cls
        return cls

class ResourcesObj(metaclass=ResourcesObjMeta):
    '''Class for resources that need to load to GPU before get into the main loop'''

    # region cls properties
    _Format:str = None # format of the data, e.g. "obj" for "Mesh_OBJ"
    '''override this property to specify the format of the mesh data, e.g. "obj"'''
    _LoadOrder = 0
    '''override this property to specify the order of loading to GPU by ResourcesManager, the smaller the earlier'''
    _BaseName = None
    '''Override this to set the base name of the class, e.g. "Mesh" for "Mesh_OBJ"'''

    _Instances: dict = None  # {name: Obj, ...}

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
    def FindFormat(cls, format):
        return cls._Format_Clses.get(format, None)
    @classmethod
    def Format(cls):
        if cls._Format is None:
            return None
        format = cls._Format.lower().strip()
        if format.startswith('.'):
            format = format[1:]
        return format
    @classmethod
    def SupportedFormats(cls):
        return cls._Format_Clses.keys()
    def __class_getitem__(cls, item):
        return cls.FindFormat(item)
    @classmethod
    def Find(cls, name):
        if cls._Instances is None:
            cls._Instances = {}
        return cls._Instances.get(name, None)
    @classmethod
    def AllInstances(cls):
        if cls._Instances is None:
            cls._Instances = {}
        return cls._Instances.values()

    @classmethod
    def _GetPathAndName(cls, path, name=None):
        '''Get proper path & name for cls.Load()'''
        if name is None:
            name = os.path.basename(path).split('.')[0]
        if cls.Find(name) is not None:
            raise ValueError(f'Resources type {cls.BaseClsName()} with name "{name}" already exists')
        return path, name
    @classmethod
    def Load(cls, path, name=None)->'ResourcesObj':
        '''
        Override this method to load the data from file.
        e.g.:
            def Load(cls, path, name=None):
                path, name = cls._GetNameAndPath(path, name)
                ... # your code here
        '''
        raise NotImplementedError
    # endregion

    def __new__(cls, name, *args, **kwargs):
        if cls._Instances is None:
            cls._Instances = {}
        if name in cls._Instances:
            obj = cls._Instances[name]
            obj.__init__ = lambda *a, **kw: None
            return obj
        else:
            obj = super().__new__(cls)
            obj._name = name
            cls._Instances[name] = obj
            return obj
    def __init__(self, name, *args, **kwargs):
        pass
    @property
    def name(self):
        return self._name
    @property
    def engine(self)->'Engine':
        return GetGlobalValue('_ENGINE_SINGLETON')

    # region abstract methods
    def sendToGPU(self):
        '''Override this to send the data to GPU'''
        print(f'sending {self.BaseClsName()}: {self.name} to GPU...')
    def clear(self):
        '''Override this to clear the data from GPU'''
        print(f'clearing {self.BaseClsName()}: {self.name} from GPU...')
    # endregion

__all__ = ['ResourcesObj', 'ResourcesObjMeta']