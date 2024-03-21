from abc import ABC, abstractmethod
from typing import Any, ClassVar, Union, Dict, Tuple

from common_utils.global_utils import GetOrCreateGlobalValue
from comfyUI.types import _get_comfy_type_name


AdaptersMap: Dict[Tuple[str, str], 'Adapter'] = GetOrCreateGlobalValue('__COMFYUI_ADAPTERS_MAP__', dict)
'''The global map for saving all adapter instances. It's a dictionary with keys as (from_type, to_type) and values as the adapter.'''

class Adapter(ABC):
    '''
    Inherit from this class to register automatic conversion between types.
    
    If you have put `From` as Any, it will be the default adapter for all types converting to `To`(But still can be overridden by specific `From`-`To` adapter).
    '''
    
    @staticmethod
    def _AllAdapterClses():
        return Adapter.__subclasses__()
    
    def __new__(cls):
        if cls.Instance is not None:
            return cls.Instance
        return super().__new__(cls)
    
    def __init__(self):
        if self.__class__.Instance is not None:
            return
        AdaptersMap[(self.From, self.To)] = self
        self.__class__.Instance = self

    
    From: ClassVar[str]
    '''from type's name'''
    To: ClassVar[str]
    '''to type's name'''
    
    Instance: ClassVar['Adapter'] = None    # type: ignore
    '''The instance of the adapter. It's a class variable.'''
    
    def __init_subclass__(cls, From: Union[type, str], To: Union[type, str]):
        cls.From = _get_comfy_type_name(From)
        cls.To = _get_comfy_type_name(To)
        cls()   # for registering & initializing the adapter
        
    @abstractmethod
    def __call__(self, val):
        '''override this method to implement the conversion.'''
        raise NotImplementedError


class AnyToStr(Adapter, From=Any, To=str):
    def __call__(self, val):
        return str(val)
    
class StrToInt(Adapter, From=str, To=int):
    def __call__(self, val: str):
        return int(val)