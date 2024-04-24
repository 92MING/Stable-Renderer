from abc import ABC, abstractmethod
from typing import Any, ClassVar, Union, Dict, Tuple, List, Type
from inspect import getmro

from common_utils.decorators import class_or_ins_property
from common_utils.global_utils import GetOrCreateGlobalValue
from comfyUI.types import get_comfy_name


AdaptersMap: Dict[Tuple[str, str], 'Adapter'] = GetOrCreateGlobalValue('__COMFYUI_ADAPTERS_MAP__', dict)
'''The global map for saving all adapter instances. It's a dictionary with keys as (from_type, to_type) and values as the adapter.'''

class Adapter(ABC):
    '''
    Inherit from this class to register automatic conversion between types.
    
    If you have put `From` as Any, it will be the default adapter for all types converting to `To`(But still can be overridden by specific `From`-`To` adapter).
    '''
    
    @staticmethod
    def AllAdapterClses()->List[Type['Adapter']]:
        return Adapter.__subclasses__()

    @staticmethod
    def _AvailableTypeConvertionInfo()->Dict[str, List[str]]:
        type_convertion_info = {}
        for (from_type_name, to_type_name) in AdaptersMap.keys():
            if from_type_name not in type_convertion_info:
                type_convertion_info[from_type_name] = []
            type_convertion_info[from_type_name].append(to_type_name)
        return type_convertion_info
    
    def __new__(cls):
        if cls.Instance is not None:
            return cls.Instance
        return super().__new__(cls)
    
    def __init__(self):
        if self.__class__.Instance is not None:
            return
        AdaptersMap[(self.FromTypeName, self.ToTypeName)] = self
        self.__class__.Instance = self

    def __init_subclass__(cls, From: Union[type, str], To: Union[type, str]):
        cls.From = From
        cls.To = To
        cls()   # for registering & initializing the adapter
        
    @abstractmethod
    def __call__(self, val):
        '''override this method to implement the conversion.'''
        raise NotImplementedError

    From: ClassVar[Union[str, type]]
    '''from type's name'''
    To: ClassVar[Union[str, type]]
    '''to type's name'''
    
    @class_or_ins_property # type: ignore
    def FromTypeName(cls_or_ins)->str:
        '''The name of the from type. It's a class or instance property.'''
        name = get_comfy_name(cls_or_ins.From)
        if name == '*':
            return 'ANY'
        return name
    
    @class_or_ins_property  # type: ignore
    def ToTypeName(cls_or_ins)->str:
        '''The name of the to type. It's a class or instance property.'''
        name = get_comfy_name(cls_or_ins.To)
        if name == '*':
            return 'ANY'
        return name
    
    Instance: ClassVar['Adapter'] = None    # type: ignore
    '''The instance of the adapter. It's a class variable.'''
    

__all__ = ['Adapter', 'AdaptersMap', ]


class AnyToStr(Adapter, From=Any, To=str):
    def __call__(self, val):
        return str(val)

class StrToInt(Adapter, From=str, To=int):
    def __call__(self, val: str):
        return int(val)

class StrToFloat(Adapter, From=str, To=float):
    def __call__(self, val: str):
        return float(val)

class StrToCombo(Adapter, From=str, To="COMBO"):
    def __call__(self, val):
        return val  # just return the value


def find_adapter(t1: Union[type, str], t2: Union[type, str]) -> Union[Adapter, None]:
    '''find proper adapter for converting from t1 to t2. Return None if not found.'''
    if t1 == '*': t1 = "ANY"
    if t2 == '*': t2 = "ANY"

    if t1 == object:
        return AdaptersMap.get(('ANY', t2), None)   # type: ignore
    
    if isinstance(t1, type):
        t2_cls_name = get_comfy_name(t2)
        t1_cls_names = [get_comfy_name(cls) for cls in getmro(t1)][:-1]   # `object` is not included
        for cls_name in t1_cls_names:
            if (cls_name, t2) in AdaptersMap:
                return AdaptersMap[(cls_name, t2_cls_name)]
        if ('ANY', t2_cls_name) in AdaptersMap: 
            return AdaptersMap[('ANY', t2_cls_name)]
        return None
    
    else:
        t1_name = get_comfy_name(t1)
        t2_name = get_comfy_name(t2)
        if (t1_name, t2_name) in AdaptersMap:
            return AdaptersMap[(t1_name, t2_name)]
        if ('ANY', t2_name) in AdaptersMap:
            return AdaptersMap[('ANY', t2_name)]
        return None

__all__.extend(['find_adapter', ])