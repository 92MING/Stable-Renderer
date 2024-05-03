from common_utils.global_utils import GetGlobalValue
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from engine.engine import Engine

class NamedObj:
    '''NamedObj is a base cls that every instance has a name. If instance with the same name is created, raise error.'''
    
    _Instances: Dict[str, 'NamedObj']

    _name: Optional[str] = None
    '''If name is not assigned, it will not be saved into the `_Instances` dict.'''

    def __new__(cls, name:Optional[str]=None, *args, **kwargs):
        '''
        For NamedObj, the first argument is always the name of the instance. If instance with the same name is created, raise error.
        Name is not case sensitive.
        '''
        if name is None:
            return super().__new__(cls)
        
        if not hasattr(cls, '_Instances') or cls._Instances is None:
            cls._Instances = {}
        if name in cls._Instances:
            ins = cls._Instances[name]
            ins.__init__ = lambda *args, **kwargs: None # prevent __init__ from being called
        else:
            ins = super().__new__(cls)
            cls._Instances[name] = ins
        return ins
    
    def __init__(self, name:str):
        self._name = name

    def __class_getitem__(cls, item):
        '''NamedObj['name'] will return the instance with name "name". Equivalent to NamedObj.GetInstance('name')'''
        return cls.Find(item)
    
    @classmethod
    def Find(cls, name: str):
        '''Find instance with name. return None if not found'''
        if cls._Instances is None:
            cls._Instances = {}
        return cls._Instances.get(name, None)
    
    @classmethod
    def AllObjs(cls):
        '''return a list of all instances'''
        if cls._Instances is None:
            cls._Instances = {}
        return list(cls._Instances.values())

    @property
    def name(self)->Optional[str]:
        return self._name
    
    @name.setter
    def name(self, new_name:str):
        if new_name == self._name:
            return
        if new_name in self._Instances:
            raise Exception(f'Instance with name {new_name} already exists, can not change name to it.')
        else:
            if self._name is not None:
                del self.__class__._Instances[self._name]
            if new_name is not None:
                self.__class__._Instances[new_name] = self
            self._name = new_name

class EngineObj:

    _Engine = None

    @property
    def engine(self)->'Engine':
        cls = self.__class__
        if cls._Engine is None:
            engine = GetGlobalValue("__ENGINE_INSTANCE__")
            if engine is None:
                raise RuntimeError("Engine is not initialized yet.")
            cls._Engine = engine
        return cls._Engine

__all__ = ['NamedObj', 'EngineObj']