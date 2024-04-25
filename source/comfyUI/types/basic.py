from torch import Tensor
from typing import (Union, TypeAlias, Annotated, Literal, Optional, List, get_origin, get_args, ForwardRef, 
                    Any, TypeVar, Generic, Type, overload, Protocol, Dict, Tuple, TYPE_CHECKING, ClassVar, Set, Callable,)
from types import NoneType
from inspect import Parameter
from enum import Enum
from pathlib import Path
from attr import attrs, attrib
from functools import partial

from common_utils.decorators import cache_property
from common_utils.type_utils import valueTypeCheck, NameCheckMetaCls, GetableFunc, DynamicLiteral
from common_utils.global_utils import is_dev_mode, is_verbose_mode
from common_utils.debug_utils import ComfyUILogger
try:
    from comfy.samplers import KSampler
except ModuleNotFoundError as e:
    if str(e) == "No module named 'comfy'":
        import sys, os
        _comfyUI_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(_comfyUI_path)
        from comfy.samplers import KSampler
    else:
        raise e    
from comfy.sd import VAE as comfy_VAE, CLIP as comfy_CLIP
from comfy.controlnet import ControlBase as comfy_ControlNetBase
from comfy.model_patcher import ModelPatcher

if TYPE_CHECKING:
    from .runtime import InferenceContext
    from .node_base import ComfyUINode

from ._utils import *


_T = TypeVar('_T')

# region basic types
NodeInputParamType: TypeAlias = Literal['required', 'optional', 'hidden']

_anno_param_meta = NameCheckMetaCls()
class AnnotatedParam(metaclass=_anno_param_meta):
    '''Annotation class for containing parameters' information for ComfyUI's node system.'''
    
    origin_type: Optional[Union[type, str, list, TypeAlias]] = None
    '''
    The origin type of the parameter. It can also be Literal for COMBO type.
    It is not necessary to specify this, but if `comfy_name` is not specified, this is required.
    '''
    extra_params: Dict[str, Any]
    '''
    The extra parameters for the type, e.g. min, max, step for INT type, etc.
    This is usually for building special param setting in ComfyUI.
    '''
    tags: Set[str]
    '''special tags, for labeling special params, e.g. lazy, ui, ...'''
    default: Any = Parameter.empty
    '''The default value for the parameter.'''
    comfy_name: Optional[str] = None
    '''Force to return this name when registering the type. It is not necessary to specify this.'''
    
    # for node input/return params
    param_name: Optional[str] = None
    '''The name of the parameter. It is not necessary to specify this, but it is recommended to specify this.'''
    param_type: Optional["NodeInputParamType"] = None
    '''The input type of the param. `None` is available for return types.'''
   
    value_formatter: Optional[Callable[[Any], Any]] = None
    '''
    The convertor for changing the input type to the real value. E.g. Tensor(input) -> LATENT(before calling the node)
    It is not necessary to specify this.
    '''
    inner_type: Optional[Union[type, str]] = None
    '''The inner type of this param type. This is for special types, e.g. Array'''
    
    def __str__(self):
        return str(self._comfy_name)
    
    def __init__(self,
                 origin: Optional[Union[type, str, TypeAlias, Parameter, 'AnnotatedParam']],
                 extra_params: Optional[dict] = None,
                 tags: Optional[Union[List[str], Tuple[str, ...], Set[str]]] = None,
                 default: Optional[Any] = Parameter.empty,
                 param_name: Optional[str] = None,
                 param_type: Optional["NodeInputParamType"] = None,
                 comfy_name: Optional[str] = None,
                 value_formatter: Optional[Callable[[Any], Any]] = None,
                 inner_type: Optional[Union[type, TypeAlias, str]] = None):
        '''
        Args:
            - origin: The origin type of the parameter. If you are using AnnotatedParam as type hint, 
                      values will be inherited from the origin type, and all other parameters will be used to update/extending the origin type.
                      You can pass parameter directly, information will be extracted from it.
            - extra_params: The extra parameters for the type, e.g. min, max, step for int type, etc.
            - tags: special tags, for labeling special params, e.g. lazy, ui, array, ...
            - default: The default value for the parameter.
            - param_name: The name of the parameter. It is not necessary to specify this, but it is recommended to specify this.
            - param_type: The input type of the param. `None` is available for return types.
        '''
        
        # checkings & type tidy up
        if param_type is not None and not valueTypeCheck(param_type, NodeInputParamType):   # type: ignore
            raise TypeError(f'Unexpected type for param_type: {param_type}. It should be one of {NodeInputParamType}.')
        
        if isinstance(origin, GetableFunc):
            try:
                origin = origin()   # type: ignore
            except TypeError as e:   # missing parameters
                raise e
        
        tags = tags or []
        if not isinstance(tags, list):
            tags = list(tags)
        extra_params = extra_params or {}
        param: Optional[Parameter] = None
        
        # for case looks like `Annotated[int, AnnotatedParam(int, ...)]`, take out the AnnotatedParam
        origin_origin = get_origin(origin)
        if origin_origin:
            
            if origin_origin == Annotated:
                origin_args = get_args(origin)
                if len(origin_args)==2:
                    if isinstance(origin_args[1], AnnotatedParam):  # e.g. Annotated[int, AnnotatedParam(int, ...)]
                        origin = origin_args[1]
                    elif isinstance(origin_args[1], str):   # e.g. Annotated[int, 'x']  to annotate the param name
                        origin = origin_args[0]
                        param_name = param_name if param_name is not None else origin_args[1]
                    else:
                        raise TypeError(f'Unexpected type for Annotated: {origin_args[1]}. It should be AnnotatedParam or str.')
                else:
                    raise TypeError(f'Annotated should only have 2 args, but got {origin_args}')
            
            elif origin_origin == Union:
                origin_args = get_args(origin)
                if not (len(origin_args)==2 and (NoneType in origin_args)):
                    raise TypeError(f'Unexpected type for Union: {origin_args}. Only Support Optional type.')
                origin = origin_args[0] if origin_args[0] != NoneType else origin_args[1]
                param_type = 'optional' # force to optional type
                
            elif origin_origin == list:
                origin = get_args(origin)[0]
                tags.append('list') # treat as array
            
            elif origin_origin == tuple:
                origin_args = get_args(origin)
                if not (len(origin_args) == 2 and origin_args[1] == Ellipsis):
                    raise TypeError(f'Unexpected type for Tuple: {origin_args}. Only Support Tuple[type, ...].')
                origin = origin_args[0]
                tags.append('list') # treat as array also
                            
        if isinstance(origin, Parameter):
            param = origin
            annotation = origin.annotation
            if annotation == Parameter.empty:   # no annotation
                origin = Any    # type: ignore
            else:                               # has annotation
                anno_origin = get_origin(annotation)
                if anno_origin: # special types
                    anno_args = get_args(annotation)
                    
                    if anno_origin == Annotated:
                        if len(anno_args) != 2:
                            raise TypeError(f'Annotated should only have 2 args, but got {anno_args}')
                        if isinstance(anno_args[1], AnnotatedParam):    # e.g. Annotated[int, AnnotatedParam(int, ...)]
                            origin = anno_args[1]
                        elif isinstance(anno_args[1], str): # e.g. Annotated[int, 'x']  to annotate the param name
                            origin = anno_args[0]
                            param_name = param_name if param_name is not None else anno_args[1]
                    
                    elif anno_origin in (Any, Literal):
                        origin = annotation
                        
                    elif anno_origin == Union:
                        if not (len(anno_args)==2 and (NoneType in anno_args)):
                            raise TypeError(f'Unexpected type for Union: {anno_args}. Only Support Optional type.')
                        origin = anno_args[0] if anno_args[0] != NoneType else anno_args[1]
                        param_type = 'optional' # force to optional type
                        
                    elif anno_origin == list:
                        origin = anno_args[0]
                        tags.append('list') # treat as array
                    
                    elif anno_origin == tuple:
                        if not (len(anno_args) == 2 and anno_args[1] == Ellipsis):
                            raise TypeError(f'Unexpected type for Tuple: {anno_args}. Only Support Tuple[type, ...].')
                        origin = anno_args[0]
                        tags.append('list') # treat as array also
                        
                    else:
                        raise TypeError(f'Unexpected type for Annotated: {anno_args[1]}. It should be AnnotatedParam or str.')
                else:   # normal types
                    origin = annotation
            
            default = default if default != Parameter.empty else param.default
            param_name = param_name if param_name is not None else param.name

        # not elif, cuz origin may turns into AnnotatedParam from above
        if isinstance(origin, AnnotatedParam):
            annotation = origin
            
            origin = origin.origin_type # type: ignore
            
            extra_params.update(annotation.extra_params)
            tags.extend(annotation.tags or [])
            
            default = default if default !=Parameter.empty else annotation.default
            param_type = param_type if param_type is not None else annotation.param_type
            param_name = param_name if param_name is not None else annotation.param_name
            comfy_name = comfy_name if comfy_name is not None else annotation.comfy_name
            value_formatter = value_formatter if value_formatter is not None else annotation.value_formatter
            inner_type = inner_type if inner_type is not None else annotation.inner_type
        
        if isinstance(origin, (list, tuple)): # sequence of values, means COMBO type
            for val in origin:
                if not isinstance(val, (int, str, float, bool)):
                    raise TypeError(f'Unexpected type in COMBO type: {val}. It should be one of int, str, float, bool.')
            origin = DynamicLiteral(*origin)  # type: ignore
        
        if isinstance(origin, ForwardRef):
            origin = origin.__forward_arg__.upper() # type name
        
        self.origin_type = origin
        self.extra_params = extra_params
        self.tags = set(tags)
        self.default = default
        self.param_name = param_name
        self.comfy_name = comfy_name
        if not self.comfy_name:
            if hasattr(origin, '__ComfyName__') and isinstance(origin.__ComfyName__, str):  # type: ignore
                self.comfy_name = origin.__ComfyName__  # type: ignore
        self.inner_type = inner_type
        
        if isinstance(self.origin_type, type):
            if issubclass(self.origin_type, Enum):
                def _enum_convertor(cls: Type[Enum], val: str):
                    '''convert the string to enum value.'''
                    val = val.lower()
                    for member in cls:
                        if member.name.lower() == val:
                            return member
                    raise ValueError(f'Invalid value for {cls.__name__}: {val}. It should be one of {list(cls.__members__.keys())}.')

                self.value_formatter = lambda val: _enum_convertor(self.origin_type, val)   # type: ignore
            else:
                self.value_formatter = value_formatter
                if not self.value_formatter and hasattr(self.origin_type, '__ComfyValueFormatter__'):
                    formatter = self.origin_type.__dict__['__ComfyValueFormatter__']
                    if formatter is not None:
                        if not isinstance(formatter, (classmethod, staticmethod)):
                            raise TypeError(f'__ComfyValueFormatter__ should be a class or static method, but got {formatter}.')
                        if isinstance(formatter, classmethod):
                            self.value_formatter = partial(formatter.__func__, self.origin_type)
                        else:
                            self.value_formatter = formatter
        else:
            self.value_formatter = value_formatter
        
        self.param_type = param_type
        if not self.param_type and param:
            self.param_type = get_input_param_type(param, self.tags)

    @property
    def _comfy_type(self)->Union[str, list]:
        '''
        Return the comfy type name(or list of value for COMBO type) for this param.
        For internal use only.
        '''
        if self.comfy_name:
            return self.comfy_name
        return get_comfy_type_definition(self.origin_type, self.inner_type) # type: ignore

    @property
    def _comfy_name(self)->str:
        '''Return the comfy name for this param. For internal use only.'''
        if self.comfy_name:
            return self.comfy_name
        return get_comfy_name(self.origin_type, self.inner_type)
    
    @property
    def proper_param_name(self)->Optional[str]:
        '''Return the proper param name shown on UI.'''
        if not self.param_name:
            return self.param_name
        name = self.param_name
        while name.startswith('_'):
            name = name[1:]
        return name
        
    @property
    def _comfyUI_definition(self)->tuple:
        '''return the tuple form for registration, e.g. ("INT", {...})'''
        data_dict = self.extra_params
        if self.default != Parameter.empty:
            data_dict['default'] = self.default
        if data_dict:
            return (self._comfy_type, data_dict)
        else:
            return (self._comfy_type, )

    def format_value(self, val: Any):
        '''format the value by the formatter'''
        if not self.value_formatter:
            return val
        
        need_convert = False
        if isinstance(self.origin_type, str):
            self_comfy_name = self._comfy_name
            if self_comfy_name != "COMBO":  # no need to convert combo
                val_comfy_name = get_comfy_name(val)
                if val_comfy_name != self_comfy_name:
                    need_convert = True
        else:
            if not valueTypeCheck(val, self.origin_type):   # type: ignore
                need_convert = True
        
        if need_convert:
            return self.value_formatter(val)
        return val


_CT = TypeVar('_CT', bound='ComfyValueType')
class ComfyValueType(Protocol, Generic[_CT]):    # type: ignore
    '''Protocol for hinting the special properties u can define in customized type for comfyUI.'''
    
    @classmethod
    def __ComfyValueFormatter__(cls, value)->Any:
        '''Convert the value to the real value for the type.'''
    
    def __ComfyDump__(self)->Any:
        '''You can define how this value should be dump to json when returning the response.'''
    
    @classmethod
    def __ComfyLoad__(cls: Type[_CT], data: Any)->_CT:  # type: ignore
        '''You can define how to load the data from json to the object.'''
    
    __ComfyName__: ClassVar[str]
    '''If you have set this field, it would be the comfy name for this type.'''

__all__ = ['NodeInputParamType', 'AnnotatedParam', 'ComfyValueType']
# endregion


# region primitive types
def INT(min: int=0, max: int = 0xffffffffffffffff, step:int=1, display: Literal['number', 'slider', 'color']='number')->Annotated:
    '''
    This is the int type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: INT(0, 10, 2)=0): ...
    
    Note:
        You can also label INT with `INT[0, 10, 2]` to put the parameters. It is a alias for `INT(0, 10, 2)`.
        
        You can still use `int` for type annotation.
    
    Args:
        - min: The minimum value.
        - max: The maximum value.
        - step: The step value.
        - display: The display type. It should be one of "number", "slider", "color".
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider', 'color'):
        raise ValueError(f'Invalid value for display: {display}. It should be one of "number", "slider", "color".')
    data = {"min": min, "max": max, "step": step, "display": display}
    return Annotated[int, AnnotatedParam(origin=int, extra_params=data, comfy_name='INT')]
globals()['INT'] = GetableFunc(INT)    # trick for faking IDE to believe INT is a function

def FLOAT(min=0, max=100, step=0.01, round=0.01, display: Literal['number', 'slider']='number')->Annotated:
    '''
    This is the float type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: FLOAT(0, 10, 2)=0): ...
    
    Notes:
        You can also label FLOAT with `FLOAT[0, 10, 2]` to put the parameters. It is a alias for `FLOAT(0, 10, 2)`.
        
        You can still use `float` for type annotation.
    
    Args:
        - min: The minimum value.
        - max: The maximum value.
        - step: The step value.
        - round: The round value.
        - display: The display type. It should be one of "number", "slider".
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider'):
        raise ValueError(f'Invalid value for display: {display}, It should be one of "number", "slider".')
    data = {"min": min, "max": max, "step": step, "round": round, "display": display}
    return Annotated[float, AnnotatedParam(origin=float, extra_params=data, comfy_name='FLOAT')]
globals()['FLOAT'] = GetableFunc(FLOAT)    # trick for faking IDE to believe FLOAT is a function

def STRING(multiline=False, forceInput: bool=False, dynamicPrompts: bool=False)->Annotated:
    '''
    This is the str type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: STRING(multiline=True)=""): ...
    
    Notes:
        You can also label STRING with `STRING[True, True, False]` to put the parameters. It is a alias for `STRING(True, True, False)`.
        
        You can still use `str` for type annotation.
        
    Args:
        - multiline: Whether to accept multiline input.
        - forceInput: Whether to force input.
        - dynamicPrompts: Whether to use dynamic prompts on UI
    '''
    data = {"multiline": multiline, "forceInput": forceInput, "dynamicPrompts": dynamicPrompts}
    return Annotated[str, AnnotatedParam(origin=str, extra_params=data, comfy_name='STRING')]
globals()['STRING'] = GetableFunc(STRING)    # trick for faking IDE to believe STRING is a function

def BOOLEAN(label_on: str='True', label_off: str='False')->Annotated:
    '''
    This is the bool type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: BOOLEAN(label_on="Yes", label_off="No")=True): ...
    
    Note:
        You can also label BOOLEAN with `BOOLEAN['Yes', 'No']` to put the parameters. It is a alias for `BOOLEAN('Yes', 'No')`.
    
        You can still use `bool` for type annotation.
    
    Args:
        - label_on: The text label for True on UI.
        - label_off: The text label for False on UI.
    '''
    data = {"label_on": label_on, "label_off": label_off}
    return Annotated[bool, AnnotatedParam(origin=bool, extra_params=data, comfy_name='BOOLEAN')]
globals()['BOOLEAN'] = GetableFunc(BOOLEAN)    # trick for faking IDE to believe BOOLEAN is a function

def PATH(accept_types: Union[List[str], str, Tuple[str, ...], None] = None,
         accept_folder: bool = False)->Annotated:
    '''
    Path type. Its actually string, but this will makes ComfyUI to create a button for selecting file/folder.
    
    Note:
        You can also label PATH with `PATH[['.jpg', '.png', '.jpeg', '.bmp', 'image/*'], True]` to put the parameters. It is a alias for `PATH(['.jpg', '.png', '.jpeg', '.bmp', 'image/*'], True)`.
        
        You can still use `str` for type annotation.
    
    Args:
        - accept_types: The file types to accept. e.g. ['.jpg', '.png', '.jpeg', '.bmp'] or 'image/*'.
                        Default = '*' (accept all types)
        - accept_folder: Whether to accept folder.
    '''
    if not accept_types:
        accept_types = '*'
    else:
        if isinstance(accept_types, (list, tuple)):
            accept_types = ','.join(accept_types)
    return Annotated[Path, AnnotatedParam(origin=Path, 
                                         extra_params={'accept_types': accept_types, 'accept_folder': accept_folder}, 
                                         comfy_name='PATH',
                                         value_formatter=Path)]
globals()['PATH'] = GetableFunc(PATH)    # trick for faking IDE to believe PATH is a function

__all__.extend(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'PATH'])
# endregion


# region common types
class LATENT(dict):
    '''Special type for wrapping the origin `LATENT` type.'''
    
    @property
    def samples(self)-> Optional[Tensor]:
        if 'samples' in self:
            return self['samples']
        return None
    
    @property
    def noise_mask(self)-> Optional[Tensor]:
        if 'noise_mask' in self:
            return self['noise_mask']
        return None
    
    @property
    def batch_index(self)-> Optional[List[int]]:
        if 'batch_index' in self:
            return self['batch_index']
        return None
    
    @classmethod
    def __ComfyValueFormatter__(cls, value):    # for converting the normal dict type to LATENT
        return cls(value)

CLIP: TypeAlias = Annotated[comfy_CLIP, AnnotatedParam(origin=comfy_CLIP, comfy_name='CLIP')]
'''type hint for ComfyUI's built-in type `CLIP`.'''

VAE: TypeAlias = Annotated[comfy_VAE, AnnotatedParam(origin=comfy_VAE, comfy_name='VAE')]
'''type hint for ComfyUI's built-in type `VAE`.'''

CONTROL_NET: TypeAlias = Annotated[comfy_ControlNetBase, AnnotatedParam(origin=comfy_ControlNetBase, comfy_name='CONTROL_NET')]
'''
Type hint for ComfyUI's built-in type `CONTROL_NET`.
This type includes all control networks in ComfyUI, including `ControlNet`, `T2IAdapter`, and `ControlLora`.
'''

MODEL: TypeAlias = Annotated[ModelPatcher, AnnotatedParam(origin=ModelPatcher, comfy_name='MODEL')]
'''type hint for ComfyUI's built-in type `MODEL`.'''

IMAGE: TypeAlias = Annotated[Tensor, AnnotatedParam(origin=Tensor, comfy_name='IMAGE')]
'''Type hint for ComfyUI's built-in type `IMAGE`.
When multiple imgs are given, they will be concatenated as 1 tensor by ComfyUI.'''

MASK: TypeAlias = Annotated[Tensor, AnnotatedParam(origin=Tensor, comfy_name='MASK')]
'''Type hint for ComfyUI's built-in type `MASK`.
Similar to `IMAGE`, when multiple masks are given, they will be concatenated as 1 tensor by ComfyUI.'''

CONDITIONING: TypeAlias = Annotated[
    List[Tuple[Tensor, Dict[str, Any]]], 
    AnnotatedParam(origin=List[List[Tuple[Tensor, Dict[str, Any]]]], comfy_name='CONDITIONING')
]
"""
The Conditioning datatype is a list of tuples, where the first element in the 
tuple is the regular Tensor output from a node, and the second element is a dictionary containing
other non-tensor outputs from the node. Refer to the docstrings of convert_cond() for more information.
An example structure of conditioning
[
    [cond_tensor_a, {"some_output_a": Any}],
    [cond_tensor_b, {"some_output_b": Any}], 
    ...
] 
"""

COMFY_SCHEDULERS: TypeAlias = DynamicLiteral(*KSampler.SCHEDULERS) # type: ignore
'''Literal annotation for choosing comfyUI's built-in schedulers.'''

COMFY_SAMPLERS: TypeAlias = DynamicLiteral(*KSampler.SAMPLERS) # type: ignore
'''Literal annotation for choosing comfyUI's built-in samplers.'''

__all__.extend(['LATENT', 'VAE', 'CLIP', 'CONTROL_NET', 'MODEL', 'IMAGE', 'MASK', 'CONDITIONING', 'COMFY_SCHEDULERS', 'COMFY_SAMPLERS',])
# endregion


# region return types
def Named(tp: Union[type, TypeAlias, AnnotatedParam], return_name: str):
    '''Type hint for specifying a named return value.'''
    return Annotated[tp, return_name]   # type: ignore
globals()['Named'] = GetableFunc(Named)    # trick for faking IDE to believe ReturnType is a function

class UI(Generic[_T]):
    '''
    The class for annotation the return value should be shown on comfy's web UI.
    The final data dict will be like:
        {ui":{_ui_name: _value, ...}, result:(value1, ..), extra_params1: {...}, extra_params2: {...}}
    
    # TODO: Since the ui return design of ComfyUI is too messy(and only support some types), 
    # TODO: we should do a general solution which returns HTML directly from here, and shows on the web UI.
    '''
    
    def __class_getitem__(cls, tp: Type[_T]):
        '''Annotation for return value'''
        return Annotated[tp, AnnotatedParam(origin=tp, tags=['ui'])]

    _ui_name: str
    '''the main key for putting in the UI return dict's "ui"'''    
    _value: _T
    '''The real return value'''
    _params: Dict[str, Any]
    '''Those params inside the "ui" dict.'''
    _extra_params: Dict[str, Any]
    '''extra params for passing to comfyUI's ui return dict.'''
    
    @property
    def ui_name(self)->str:
        '''The main key for putting in the UI return dict's "ui".'''
        return self._ui_name
    @property
    def value(self)->Any:
        '''The real return value.'''
        return self._value
    @property
    def extra_params(self)->Dict[str, Any]:
        '''Extra params for passing to comfyUI's ui return dict.'''
        return self._extra_params
    
    def __init__(self, ui_name:str, value: _T, **params):
        self._ui_name = ui_name
        self._value = value
        self._params = params
        self._extra_params = {}
        
    def add_extra_param(self, key, value):
        '''add extra param for the return dict.'''
        self._extra_params[key] = value

class UIImage(UI[IMAGE]):
       
    def __init__(self, 
                 value: IMAGE, 
                 filename: str, 
                 subfolder:str,
                 type: Literal['output', 'temp']='output', 
                 animated: bool=False):
        '''
        Args:
            - value: the original image
            - filename: the filename of the image(under the `output` folder)
            - subfolder: the subfolder of the image(under the `output` folder)
            - type: (output/temp) `output` means the image is saved under the output folder, `temp` means the image is saved under the temp folder.
            - animated: whether the image is animated(e.g. gif)
            - extra_params: extra params for putting in UI return dict.
        '''
        
        super().__init__('images', value, filename=filename, type=type, subfolder=subfolder)
        self.add_extra_param('animated', animated)

class UILatent(UI[LATENT]):
    def __init__(self, value: LATENT, **params):
        super().__init__('latents', value, **params)

__all__.extend(['Named', 'UI', 'UIImage', 'UILatent'])
# endregion


# region special types
@attrs
class Lazy(Generic[_T]):
    '''
    Mark the type as lazy, for lazy loading in ComfyUI.
    This could only be used in input parameters.
    '''
    
    def __class_getitem__(cls, tp: Type[_T]):
        '''Support the type hint syntax, e.g. Lazy[int]'''
        real_type = tp.origin_type if isinstance(tp, AnnotatedParam) else tp
        return Annotated[real_type, AnnotatedParam(tp, tags=['lazy'])]
    
    from_node_id: str = attrib()
    '''from which node id'''
    from_output_slot: int = attrib()
    '''from which output slot'''
    to_node_id: str = attrib()
    '''to which node id'''
    to_param_name: str = attrib()
    '''to which param name'''
    context: "InferenceContext" = attrib()
    '''which context this lazy value belongs to'''
    
    _gotten: bool = attrib(default=False)
    '''whether the value has been gotten.'''
    _value: _T = attrib(default=None)
    '''the real value of the lazy type. It will only be resolved when it is accessed.'''
    
    @cache_property  # type: ignore
    def from_node_cls_name(self)->str:
        prompt = self.context.prompt
        return prompt[self.from_node_id]['class_type']
    
    @cache_property  # type: ignore
    def from_node_cls(self)->Type["ComfyUINode"]:
        return get_node_cls_by_name(self.from_node_cls_name)  # type: ignore
    
    @cache_property  # type: ignore
    def to_node_cls_name(self)->str:
        prompt = self.context.prompt
        return prompt[self.to_node_id]['class_type']
    
    @cache_property  # type: ignore
    def to_node_cls(self)->Type["ComfyUINode"]:
        return get_node_cls_by_name(self.to_node_cls_name)    # type: ignore
    
    @cache_property # type: ignore
    def to_type_name(self)->str:
        if not hasattr(self, '_to_type_name'):
            from ._utils import get_comfy_node_input_type
            self._to_type_name = get_comfy_node_input_type(self.to_node_cls, self.to_param_name)
        return self._to_type_name
    
    @cache_property # type: ignore
    def is_list_val(self)->bool:
        return check_input_param_is_list_type(self.to_param_name, self.to_node_cls)
    
    @cache_property # type: ignore
    def from_val_type(self)->str:
        from_node_type = self.from_node_cls
        return from_node_type.RETURN_TYPES[self.from_output_slot]
    
    @property
    def value(self)->_T:   # type: ignore
        if not self._gotten:
            if is_dev_mode() and is_verbose_mode():
                ComfyUILogger.debug(f"Lazy[{self.to_type_name}] is getting the value from node: {self.from_node_cls_name}({self.from_node_id}) for {self.to_node_cls_name}({self.to_node_id})'s {self.to_param_name}, slot: {self.from_output_slot}...")
            
            from comfyUI.adapters import find_adapter
            from .runtime import NodePool
            from comfyUI.execution import PromptExecutor
            pool = NodePool()
            executor: PromptExecutor = PromptExecutor.Instance  # type: ignore
            
            from_node_id = self.from_node_id
            from_node = pool.get_node(from_node_id, self.from_node_cls_name, create_new=True, raise_err=False)
            if not from_node:
                raise ValueError(f"Node `{self.from_node_cls_name}`({from_node_id}) not found.")
            slot_index = self.from_output_slot
            to_type_name = self.to_type_name
            is_list_val = self.is_list_val
            
            context = self.context
            
            if from_node_id in context.outputs and from_node_id in context.executed_node_ids:
                if is_dev_mode() and is_verbose_mode():
                    ComfyUILogger.debug(f"Lazy[{self.to_type_name}] got its value from context's executed output.")
                val = context.outputs[from_node_id][slot_index]
            else:
                if is_dev_mode() and is_verbose_mode():
                    ComfyUILogger.debug(f"Lazy[{self.to_type_name}] is getting the value from node's output...")
                context.current_node_id = from_node_id
                executor._recursive_execute(context)    # continue execution
                val = context.outputs[from_node_id][slot_index]
            
            if not is_list_val and isinstance(val, list) and len(val) == 1:
                val = val[0]
                
            if adapter := find_adapter(self.from_val_type, to_type_name):
                val = adapter(val)
            
            self._value = val   # type: ignore
            self._gotten = True
            
        return self._value  # type: ignore
    
class Array(list, Generic[_T]):
    '''
    Mark the type as array, to allow the param accept multiple values and pack as a list.
    This could only be used in input parameters.
    
    TODO: this type is not yet finished
    '''
    
    def __class_getitem__(cls, tp: Union[type, TypeAlias, AnnotatedParam]) -> Annotated:
        '''
        Mark the type as array, to allow the param accept multiple values and pack as a list.
        This could only be used in input parameters.
        '''
        real_type = tp.origin_type if isinstance(tp, AnnotatedParam) else tp
        return Annotated[real_type, AnnotatedParam('ARRAY', inner_type=tp, tags=['list'])]    # type: ignore

    @classmethod
    def __ComfyValueFormatter__(cls, value):
        '''Convert the value to array type.'''
        raise NotImplementedError   # TODO: implement this


__all__.extend(['Lazy', 'Array'])
# endregion