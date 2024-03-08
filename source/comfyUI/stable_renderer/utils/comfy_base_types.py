from torch import Tensor
from typing import (Union, TYPE_CHECKING, TypeAlias, Protocol, Annotated, Literal, Optional, List, Sequence,
                    get_origin, get_args, ForwardRef, Tuple, Any, runtime_checkable)
from inspect import Parameter
from enum import Enum
from dataclasses import dataclass
from comfy.samplers import KSampler

if TYPE_CHECKING:
    from comfy.sd import VAE as comfy_VAE, CLIP as comfy_CLIP
    from comfy.controlnet import (ControlNet as comfy_ControlNet, T2IAdapter as comfy_T2IAdapter,
                                      ControlLora as comfy_ControlLora)
    from comfy.model_base import BaseModel


# region comfyUI primitive types
@runtime_checkable
class _ComfyUIBuiltInPrimitive(Protocol):
    def to_dict(self, default=None)->dict: ...

@dataclass
class IntType:
    '''This is the int type annotation for containing more information.'''
    min: int
    max: int
    step: int
    display: Literal['number', 'slider'] = 'number'
    
    __ComfyUI_Name__ = "INT"
    
    def to_dict(self, default=None):
        data = {"min": self.min, "max": self.max, "step": self.step, "display": self.display}
        if default is not None and default != Parameter.empty:  
            data['default'] = default
        return data
        
def INT(min=0, max=0xffffffffffffffff, step=1, display: Literal['number', 'slider', 'color']='number')->Annotated:
    '''
    This is the int type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: INT(0, 10, 2)=0): ...
    
    You can still use `int` for type annotation.
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider'):
        raise ValueError(f'Invalid value for display: {display}')
    return Annotated[int, IntType(min, max, step, display)]


@dataclass
class FloatType:
    min: float
    max: float
    step: float = 0.01
    round: float = 0.001
    display: Literal['number', 'slider'] = 'number'
    
    __ComfyUI_Name__ = "FLOAT"
    
    def to_dict(self, default=None):
        data = {"min": self.min, "max": self.max, "step": self.step, "round": self.round, "display": self.display}
        if default is not None and default != Parameter.empty:
            data['default'] = default
        return data
    
def FLOAT(min=0, max=100, step=0.01, round=0.001, display: Literal['number', 'slider']='number')->Annotated:
    '''
    This is the float type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: FLOAT(0, 10, 2)=0): ...
    
    You can still use `float` for type annotation.
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider'):
        raise ValueError(f'Invalid value for display: {display}')
    return Annotated[float, FloatType(min, max, step, round, display)]


@dataclass
class StringType:
    '''This is the str type annotation for containing more information.'''
    multiline: bool = False
    
    __ComfyUI_Name__ = "STRING"
    
    def to_dict(self, default=None):
        data = {"multiline": self.multiline}
        if default is not None and default != Parameter.empty:
            data['default'] = default
        return data
    
def STRING(multiline=False)->Annotated:
    '''
    This is the str type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: STRING(multiline=True)=""): ...
    
    You can still use `str` for type annotation.
    '''
    return Annotated[str, StringType(multiline)]

__all__ = ['INT', 'FLOAT', 'STRING']
# endregion


# region advanced basic types
class LATENT:
    '''Special type for wrapping the origin `LATENT` type.'''
    
    __ComfyUI_Name__ = "LATENT"
    
    def __init__(self, origin_dict):
        self.origin_dict = origin_dict
        
    def __getitem__(self, key):
        return self.origin_dict[key]
    
    @property
    def samples(self)-> Optional[Tensor]:
        return self.origin_dict.get('samples', None)
    
    @property
    def noise_mask(self)-> Optional[Tensor]:
        return self.origin_dict.get('noise_mask', None)
    
    @property
    def batch_index(self)-> Optional[List[int]]:
        return self.origin_dict.get('batch_index', None)

COMFY_SCHEDULERS = Literal[""]
COMFY_SCHEDULERS.__args__ = tuple(KSampler.SCHEDULERS)  # type: ignore
# override the __args__ here cuz python 3.10 doesn't support Literal[*KSampler.SCHEDULERS]

COMFY_SAMPLERS = Literal[""]
COMFY_SAMPLERS.__args__ = tuple(KSampler.SAMPLERS)  # type: ignore
# override the __args__ here cuz python 3.10 doesn't support Literal[*KSampler.SAMPLERS]

CLIP: TypeAlias = Annotated['comfy_CLIP', 'CLIP']
'''type hint for ComfyUI's built-in type `CLIP`.'''
setattr(CLIP, '__ComfyUI_Name__', 'CLIP')


VAE: TypeAlias = Annotated['comfy_VAE', 'VAE']
'''type hint for ComfyUI's built-in type `VAE`.'''
setattr(VAE, '__ComfyUI_Name__', 'VAE')


CONTROL_NET: TypeAlias = Annotated[Union['comfy_ControlNet', 'comfy_T2IAdapter', 'comfy_ControlLora'], 'CONTROL_NET']
'''
Type hint for ComfyUI's built-in type `CONTROL_NET`.
This type includes all control networks in ComfyUI, including `ControlNet`, `T2IAdapter`, and `ControlLora`.
'''
setattr(CONTROL_NET, '__ComfyUI_Name__', 'CONTROL_NET') 


MODEL: TypeAlias = Annotated['BaseModel', 'MODEL']
'''type hint for ComfyUI's built-in type `MODEL`.'''
setattr(MODEL, '__ComfyUI_Name__', 'MODEL')


IMAGE: TypeAlias = Annotated[Tensor, 'IMAGE']
'''
type hint for ComfyUI's built-in type `IMAGE`.
When multiple imgs are given, they will be concatenated as 1 tensor by ComfyUI.
'''
setattr(IMAGE, '__ComfyUI_Name__', 'IMAGE')


MASK: TypeAlias = Annotated[Tensor, 'MASK']
'''
type hint for ComfyUI's built-in type `MASK`.
Similar to `IMAGE`, when multiple masks are given, they will be concatenated as 1 tensor by ComfyUI.
'''
setattr(MASK, '__ComfyUI_Name__', 'MASK')



__all__.extend(['LATENT', 'COMFY_SCHEDULERS', 'COMFY_SAMPLERS', 'VAE', 'CLIP', 'CONTROL_NET', 'MODEL', 'IMAGE', 'MASK'])
# endregion


# region UI Types
@dataclass
class UIType:
    '''
    Return types which specific for showing on UI.
    Currently this only works for these types (since I only found these types in comfyUI's nodes):
        - IMAGE
        - LATENT
    '''
    origin_type: Literal['IMAGE', 'LATENT']
    '''The origin type of this UI type.'''
    animated: bool = False
    '''Only works for IMAGE type. Whether this image is animated or not.'''
    
    def to_dict(self, value):
        '''return the val dict for ComfyUI'''
        key = 'images' if value.origin_type == 'IMAGE' else 'latents'
        data = {key: value}
        if self.origin_type == 'IMAGE':
            data['animated'] = (self.animated, )
        if isinstance(value, Sequence):
            data['result'] = tuple(value)
        else:
            data['result'] = (value, )
        return data

UI_IMAGE = Annotated['IMAGE', UIType('IMAGE', False)]
'''Specify the return image should be shown in UI.'''

UI_ANIMATION = Annotated['IMAGE', UIType('IMAGE', True)]
'''Specify the return animated image should be shown in UI.'''

UI_LATENT = Annotated['LATENT', UIType('LATENT')]
'''Specify the return latent should be shown in UI.'''

__all__.extend(['UI_IMAGE', 'UI_ANIMATION', 'UI_LATENT'])
# endregion


_SPECIAL_TYPE_NAMES = {
    str: "STRING",
    bool: "BOOLEAN",
    int: "INT",
    float: "FLOAT",
}

def get_type_name_for_comfyUI(tp: Union[type, str, ForwardRef, TypeAlias])->str:  # type: ignore
    if tp in _SPECIAL_TYPE_NAMES:
        return _SPECIAL_TYPE_NAMES[tp]   # type: ignore
    elif get_origin(tp) == Annotated:
        return get_type_name_for_comfyUI(get_args(tp)[0])
    elif hasattr(tp, '__ComfyUI_Name__'):
        return tp.__ComfyUI_Name__  # type: ignore
    elif isinstance(tp, str): # string annotation
        return tp.upper() 
    elif isinstance(tp, ForwardRef):
        return tp.__forward_arg__.upper()
    elif tp == Any or tp == Parameter.empty:
        return "ANY"
    
    if hasattr(tp, '__qualname__'):
        return tp.__qualname__.split('.')[-1].upper()
    elif hasattr(tp, '__name__'):
        return tp.__name__.split('.')[-1].upper()
    else:
        raise TypeError(f'Cannot get type name for {tp}')

def get_type_info_for_comfyUI(param: Parameter)->Union[Tuple[str, ], Tuple[list, ], Tuple[str, dict]]:
    '''
    Return the required type information dict for comfyUI's node system.
    
    E.g. 
       - x:int=1 -> ("INT", {"default": 1})
       - x: INT(0, 10, 2)=0 -> ("INT", {"min": 0, "max": 10, "step": 2, "default": 0})
       - x: Literal[1, 2, 3] -> [1, 2, 3]   # combo doesn't need type name, just return the list
       - x: Enum -> list(Enum.__members__.keys())  # enum type is also treated as combo, the keys are the option names
       - x: A = None -> ("A", )  # A is a custom class
    '''
    anno = param.annotation
        
    if origin:= get_origin(anno):
        if origin == Literal:
            return (list(anno.__args__), )
        elif origin == Annotated:
            annotated_info = get_args(anno)
            if len(annotated_info) != 2:
                raise TypeError(f'Unexpected Annotated type: {anno}')
            anno_type, anno_info = annotated_info
            type_name = get_type_name_for_comfyUI(anno_type)
            if isinstance(anno_info, _ComfyUIBuiltInPrimitive):
                return (type_name, anno_info.to_dict(param.default))
            else:
                if param.default != Parameter.empty:
                    return (type_name, {"default": param.default})
                return (type_name, )
        else:
            raise TypeError(f'Unexpected type annotation: {anno}')
    
    elif type(anno) == type and issubclass(anno, Enum):
        return (list(anno.__members__.keys()), )
    
    elif anno in (Any, Parameter.empty):    # Any type or no type annotation
        return ("ANY", )
    
    else:   # not special type, just return the type name.
        type_name = get_type_name_for_comfyUI(anno)
        if param.default != Parameter.empty:
            return (type_name, {"default": param.default})
        elif anno in (int, float, str):
            return (type_name, {})
        else:
            return (type_name, )

def get_return_type_and_names_for_comfyUI(return_type: Union[type, TypeAlias])->Union[Tuple[Tuple[str, ...], Optional[Tuple[str, ...]]], UIType]:
    '''
    Return the `RETURN_TYPES` and `RETURN_NAMES` for comfyUI's node system.
    
    e.g. 
        - def f(x: int)->int:...  ---> ("INT", ), None  # no return name
        - def f(x: int)->Annotated[int, 'ABC']:...  ---> ("INT", ), ("ABC", ) # return name is 'ABC'
        - def f(x: int)->Tuple[int, int]:...  ---> ("INT", "INT"), None  # no return name, 2 return values
        - def f(x: int)->Tuple[Annotation[int, 'A'], Annotation[int, 'B']]:...  ---> ("INT", "INT"), ("A", "B")  # 2 return names
    
    Special case: 
        when you specify the return type as `UI_IMAGE`/`UI_ANIMATION`/`UI_LATENT`, it will return the UIType directly.
    '''
    if origin:= get_origin(return_type):
        if origin == Annotated:
            annotated_info = get_args(return_type)
            if len(annotated_info) != 2:
                raise TypeError(f'Unexpected Annotated type: {return_type}')
            ret_type = annotated_info[0]
            ret_name_or_ui_type = annotated_info[1]
            if isinstance(ret_name_or_ui_type, str):
                return (get_type_name_for_comfyUI(ret_type), ), (ret_name_or_ui_type, )
            elif isinstance(ret_name_or_ui_type, UIType):
                return ret_name_or_ui_type
            else:
                return (get_type_name_for_comfyUI(ret_type), ), None
        elif origin == Tuple:
            ret_types = []
            ret_names = []
            for arg in get_args(origin):
                try:
                    ret_type, ret_name_or_ui_type = get_return_type_and_names_for_comfyUI(arg)
                except TypeError:   # will raise error when the arg is not a type/ UIType within other types
                    raise TypeError(f'Unexpected type annotation: {arg}')
                ret_types.extend(ret_type)
                if ret_name_or_ui_type:
                    ret_names.extend(ret_name_or_ui_type)
                else:
                    ret_names.extend([""]*len(ret_type))
            
            if all(name == "" for name in ret_names):
                return tuple(ret_types), None
            return tuple(ret_types), tuple(ret_names)
        else:
            raise TypeError(f'Unexpected type annotation: {return_type}')
    else:
        return (get_type_name_for_comfyUI(return_type), ), None



__all__.extend(['get_type_name_for_comfyUI', 'get_type_info_for_comfyUI', 'get_return_type_and_names_for_comfyUI'])