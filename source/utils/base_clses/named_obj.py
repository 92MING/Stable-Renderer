'''NamedObj is a base cls that every instance has a name. If instance with the same name is created, raise error.'''

from .cross_module_class import CrossModuleClass

class NamedObj(CrossModuleClass):
    _instances = None

    def __new__(cls, name:str, *args, **kwargs):
        '''
        For NamedObj, the first argument is always the name of the instance. If instance with the same name is created, raise error.
        Name is not case sensitive.
        '''
        name = name.lower()
        if cls._instances is None:
            cls._instances = {}
        if name in cls._instances:
            raise Exception(f'Instance with name {name} already exists')
        else:
            ins = super().__new__(cls)
            cls._instances[name] = ins
        return ins
    def __init__(self, name:str, *args, **kwargs):
        self._name = name.lower()

    @classmethod
    def GetInstance(cls, ins_name):
        '''return None if not found'''
        if cls._instances is None:
            cls._instances = {}
        return cls._instances.get(ins_name, None)

    def __class_getitem__(cls, item):
        '''NamedObj['name'] will return the instance with name "name". Equivalent to NamedObj.GetInstance('name')'''
        return cls.GetInstance(item)

    @property
    def name(self)->str:
        return self._name

__all__ = ['NamedObj']