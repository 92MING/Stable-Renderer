from uuid import uuid4
from typing import Dict, TYPE_CHECKING, Set, Optional, Type, ClassVar, TypeVar
from attr import attrs, attrib
from abc import ABC, ABCMeta, abstractmethod

from common_utils.debug_utils import EngineLogger
from common_utils.global_utils import GetOrCreateGlobalValue, GetGlobalValue, is_engine_looping, is_dev_mode
 
if TYPE_CHECKING:
    from engine.engine import Engine


def _random_str():
    return str(uuid4()).replace('-', '')

__ENGINE_RESOURCES_CLSES__ = GetOrCreateGlobalValue('__ENGINE_RESOURCES_CLSES__', dict)
'''cls_name: cls'''

__ENGINE_FORMAT_SUBCLSES__: Dict[str, Dict[str, type]] = GetOrCreateGlobalValue('__ENGINE_FORMAT_SUBCLSES__', dict)
'''{ base_cls_name: {format: cls} }'''

__NAMED_RESOURCES_OBJS__: Dict[str, Dict[str, 'ResourcesObj']] = GetOrCreateGlobalValue('__RESOURCES_OBJS__', dict)
'''containing all named resources objects, {cls_name: {name: obj}}'''

__TO_BE_LOAD_RESOURCES__: Set['ResourcesObj'] = GetOrCreateGlobalValue('__TO_BE_LOAD_RESOURCES__', set)
__TO_BE_DESTROY_RESOURCES__: Set['ResourcesObj'] = GetOrCreateGlobalValue('__TO_BE_DESTROY_RESOURCES__', set)


class ResourcesObjMeta(ABCMeta):
    
    Format: Optional[str] = None
    BaseClsName: str

    def __new__(cls, *args, **kwargs):
        cls_name = args[0]
        cls = super().__new__(cls, *args, **kwargs)
        if cls_name == 'ResourcesObj':
            return cls
        if cls.BaseClsName is not None:
            if cls.BaseClsName not in __ENGINE_FORMAT_SUBCLSES__:
                __ENGINE_FORMAT_SUBCLSES__[cls.BaseClsName] = {}
            if cls.Format is not None and cls.Format not in __ENGINE_FORMAT_SUBCLSES__[cls.BaseClsName]:
                format = cls.Format.strip().lower()
                if format.startswith('.'):
                    format = format[1:]
                __ENGINE_FORMAT_SUBCLSES__[cls.BaseClsName][format] = cls
        __ENGINE_RESOURCES_CLSES__[cls_name] = cls
        return cls

_RC = TypeVar('_RC', bound='ResourcesObj')

@attrs(eq=False, repr=False)
class ResourcesObj(ABC, metaclass=ResourcesObjMeta):
    '''Class for resources that need to load to GPU before get into the main loop'''

    # region cls properties
    Format: ClassVar[Optional[str]] = None # format of the data, e.g. "mtl" for "Material_MTL"
    '''override this property to specify the format of the mesh data, e.g. "obj"'''
    LoadOrder: ClassVar[int] = 0
    '''override this property to specify the order of loading to GPU by ResourcesManager, the smaller the earlier'''
    BaseClsName: ClassVar[str]
    '''Override this to set the base name of the class'''
    _Engine: ClassVar["Engine"]
    '''The engine instance. This is for your easy access to the engine instance. Do not override this.'''
    
    id: str = attrib(factory=_random_str, init=False)
    '''For internal use only, do not override this'''
    _name: Optional[str] = attrib(default=None, alias='name', kw_only=True)
    '''The name of the obj can be None. If this is set, it will be used for finding the obj.'''
    _immediate_load: bool = attrib(default=True, alias='immediate_load', kw_only=True)
    '''If True, the obj will be loaded immediately after created. If False, you need to call `load` manually.'''
    _destroyed: bool = attrib(default=False, init=False)
    '''internal use only, do not override this'''
    alias: Optional[str] = attrib(default=None, kw_only=True)
    '''Alias name is just for printing. It is useful when you dont wanna set the name but you want to recognize the obj easily.'''
    
    @classmethod
    def FindFormatCls(cls: Type[_RC], format: str)->Optional[Type[_RC]]:
        '''Find the sub-class that supports the given format, e.g. Mesh.FindFormat("obj") will return Mesh_OBJ.'''
        format = format.strip().lower()
        if format.startswith('.'):
            format = format[1:]
        if cls.Format is not None and cls.Format == format:
            return cls
        return __ENGINE_FORMAT_SUBCLSES__[cls.BaseClsName].get(format, None)
    
    @classmethod
    def Find(cls: Type[_RC], name: str)->Optional[_RC]:
        '''find named resources object'''
        if cls.BaseClsName not in __NAMED_RESOURCES_OBJS__:
            __NAMED_RESOURCES_OBJS__[cls.BaseClsName] = {}
        if cls == ResourcesObj:
            for _, objs in __NAMED_RESOURCES_OBJS__.items():
                obj = objs.get(name, None)
                if obj is not None:
                    return obj  # type: ignore
            else:
                return None
        return __NAMED_RESOURCES_OBJS__[cls.BaseClsName].get(name, None)    # type: ignore
    
    @classmethod
    def ClassName(cls):
        return cls.__qualname__.split('.')[-1]
    
    @classmethod
    def AllObjs(cls):
        if cls == ResourcesObj:
            return tuple(__TO_BE_DESTROY_RESOURCES__)
        elif cls.Format is not None:
            return tuple([obj for obj in __TO_BE_DESTROY_RESOURCES__ if obj.Format == cls.Format])
        else:
            return tuple([obj for obj in __TO_BE_DESTROY_RESOURCES__ if obj.BaseClsName == cls.BaseClsName])
    
    @classmethod
    def SupportedFormats(cls)->list[str]:
        '''check the supported formats of the sub-classes of this class'''
        if cls == ResourcesObj:
            raise ValueError('This method should be called on sub-classes of ResourcesObj')
        return list(__ENGINE_FORMAT_SUBCLSES__[cls.BaseClsName].keys())

    def __class_getitem__(cls, item):
        '''Allow to use cls[format] to get the sub-class that supports the given format'''
        return cls.FindFormatCls(item)

    def __attrs_post_init__(self):
        if self.name is not None:
            if self.BaseClsName not in __NAMED_RESOURCES_OBJS__:
                __NAMED_RESOURCES_OBJS__[self.BaseClsName] = {}
            __NAMED_RESOURCES_OBJS__[self.BaseClsName][self.name] = self
        if self._immediate_load:
            if is_engine_looping():
                self.load()
            else:
                __TO_BE_LOAD_RESOURCES__.add(self)
        __TO_BE_DESTROY_RESOURCES__.add(self)
    
    @property
    def name(self)->Optional[str]:
        return self._name
    
    @name.setter
    def name(self, name: str):
        if self._name is not None:
            raise ValueError('The name of the resources object can not be changed once set.')
        if self.BaseClsName not in __NAMED_RESOURCES_OBJS__:
            __NAMED_RESOURCES_OBJS__[self.BaseClsName] = {}
        if name in __NAMED_RESOURCES_OBJS__[self.BaseClsName]:
            raise ValueError(f'The name "{name}" has been used by another resources object.')
        if self._name is not None:
            __NAMED_RESOURCES_OBJS__[self.BaseClsName][name] = __NAMED_RESOURCES_OBJS__[self.BaseClsName].pop(self._name)
        self._name = name
    
    @property
    def engine(self)->'Engine':
        '''you can access the engine instance by `self.engine` easily. This is a read-only property.'''
        if not hasattr(self.__class__, '_Engine') or self.__class__._Engine is None:
            engine = GetGlobalValue('__ENGINE_INSTANCE__')
            if engine is None:
                raise ValueError('No engine instance found. You can not access `engine` before the engine is created.')
            self.__class__._Engine = engine
        return self.__class__._Engine

    @abstractmethod
    def load(self):
        '''Override this to send the data to host/GPU/...'''
        if is_dev_mode():
            EngineLogger.info(f'{self} is loading...')
    
    @property
    @abstractmethod
    def loaded(self)->bool:
        '''override this to check if the data has been loaded to host/GPU/...'''
    
    @abstractmethod
    def clear(self):
        '''Override this to clear the data from host/GPU/...'''
        if is_dev_mode():
            EngineLogger.info(f'{self} is clearing...')

    def destroy(self):
        '''destroy the this obj completely, including removing the name from the named resources objects dict.'''
        if self._destroyed:
            if self in __TO_BE_LOAD_RESOURCES__:
                __TO_BE_LOAD_RESOURCES__.remove(self)
            return
        if self.loaded:
            self.clear()
        if self.name is not None:
            __NAMED_RESOURCES_OBJS__[self.BaseClsName].pop(self.name, None)
        if self in __TO_BE_LOAD_RESOURCES__:
            __TO_BE_LOAD_RESOURCES__.remove(self)
        self._destroyed = True
        if is_dev_mode():
            EngineLogger.info(f'{self} has been destroyed.')
    

    def __del__(self):
        if self.name is not None:
            __NAMED_RESOURCES_OBJS__[self.BaseClsName].pop(self.name, None)
        self.clear()
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, ResourcesObj) and self.id == other.id

    def __repr__(self):
        name = self.name or self.alias or 'unnamed'
        return f'<{self.ClassName()}: {name} (id={self.id})>'
    
    

__all__ = ['ResourcesObj', 'ResourcesObjMeta']