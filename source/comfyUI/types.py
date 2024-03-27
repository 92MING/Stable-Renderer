from collections.abc import Iterator
from torch import Tensor
from typing import (Union, TypeAlias, Annotated, Literal, Optional, List, get_origin, get_args, ForwardRef, 
                    Any, TypeVar, Generic, Type, overload, Protocol, Dict, Tuple, TYPE_CHECKING,
                    _ProtocolMeta, runtime_checkable, Sequence, ClassVar, Set, Callable, TypeGuard)
from types import NoneType
from inspect import Parameter
from enum import Enum
from pathlib import Path
from abc import ABC, abstractclassmethod
from deprecated import deprecated

from common_utils.type_utils import valueTypeCheck, get_cls_name
from common_utils.decorators import singleton, class_property, class_or_ins_property
from comfy.samplers import KSampler
from comfy.sd import VAE as comfy_VAE, CLIP as comfy_CLIP
from comfy.controlnet import ControlBase as comfy_ControlNetBase
from comfy.model_base import BaseModel

if TYPE_CHECKING:
    from comfyUI.execution import PromptExecutor
    from comfyUI.stable_renderer.nodes.node_base import NodeBase


_T = TypeVar('_T')
_HT = TypeVar('_HT', bound='HIDDEN')

_SPECIAL_TYPE_NAMES = {
    str: "STRING",
    bool: "BOOLEAN",
    int: "INT",
    float: "FLOAT",
    Path: "PATH",
}

def _get_comfy_type_definition(tp: Union[type, str, TypeAlias], 
                               inner_type: Optional[Union[type, str]]=None)->Union[str, list]:
    '''
    return the type name or list of value for COMBO type.
    e.g. int->"INT", Literal[1, 2, 3]->[1, 2, 3]
    '''
    if inner_type is not None:
        tp_definition = _get_comfy_type_definition(inner_type)
        if not isinstance(tp_definition, str):
            raise TypeError(f'This type {tp} is not supported for inner type.')
        inner_type_definition = tp_definition
        if not isinstance(inner_type_definition, str):
            raise TypeError(f'This type {inner_type} is not supported for being as inner type.')
        return f"{tp_definition}[{inner_type_definition}]"
    
    if isinstance(tp, str): # string annotation
        return tp.upper()     # default return type name to upper case
    
    if isinstance(tp, (list, tuple)):   # COMBO type
        for t in tp:
            if not isinstance(t, (int, str, float, bool)):
                raise TypeError(f'Unexpected type in COMBO type: {tp}. It should be one of int, str, float, bool.')
        return list(tp)
    
    if tp in _SPECIAL_TYPE_NAMES:
        return _SPECIAL_TYPE_NAMES[tp]   # type: ignore
    
    if isinstance(tp, type) and issubclass(tp, Enum):
        return list(tp.__members__.keys())  # return the keys of the enum as the combo options
    
    origin = get_origin(tp)
    if origin:
        if origin == Literal:
            return [arg for arg in tp.__args__]
        elif origin == Annotated:
            args = get_args(tp)
            for arg in args[1:]:
                if isinstance(arg, AnnotatedParam):
                    return arg._comfy_type
            raise TypeError(f'Unexpected Annotated type: {tp}')
        else:
            raise TypeError(f'Unexpected type annotation: {tp}')
    
    if isinstance(tp, ForwardRef):
        return tp.__forward_arg__.upper()
    
    if tp == Any or tp == Parameter.empty:
        return "*"  # '*' represents any type in litegraph
    
    try:
        return get_cls_name(tp).upper()
    except:
        raise TypeError(f'Unexpected type: {tp}. Cannot get the type name.')

def _get_input_param_type(param: Parameter, 
                          tags: Optional[Set[str]]=None)->Literal['required', 'optional', 'hidden']:
    '''
    Get the input parameter type for registration.
    
    Args:
        * param: the parameter object
        * tags: the tags of the parameter
    
    Returns:
        * hidden: param name starts with '_', e.g. def f(_x:int)
        * optional: param default is None, e.g. def f(x:int = None)
        * required: others
    '''
    # 3 conditions for hidden
    input_param_name = param.name
    if input_param_name.startswith('_'): # 1. param name starts with '_'
        return 'hidden'
    
    anno = param.annotation
    anno_origin = get_origin(anno)
    if anno_origin:
        if anno_origin == Annotated:
            args = get_args(anno)
            if isinstance(args[0], str):
                if args[0].lower() == 'hidden': # 2. type annotation is 'hidden' or is a subclass of HIDDEN
                    return 'hidden'
            if isinstance(args[0], type):
                if issubclass(args[0], HIDDEN): 
                    return 'hidden'
    elif type(anno) == type:
        if issubclass(anno, HIDDEN):
            return 'hidden'
    elif isinstance(anno, str):
        if anno.lower() == 'hidden':
            return 'hidden'
    if tags:    # 3. tags contains 'hidden'
        if 'hidden' in tags:
            return 'hidden'
    
    if param.default != Parameter.empty:    # if has default value, it is optional
        return 'optional'
    
    return 'required'  

def _enum_convertor(cls: Type[Enum], val: str):
    val = val.lower()
    for member in cls:
        if member.name.lower() == val:
            return member
    raise ValueError(f'Invalid value for {cls.__name__}: {val}. It should be one of {list(cls.__members__.keys())}.')

def get_comfy_name(tp: Any, inner_type: Optional[Union[type, str]]=None)->str:
    '''Get the type name for comfyUI. It can be any value or type, or name'''
    type_name = _get_comfy_type_definition(tp, inner_type)
    if isinstance(type_name, list):
        type_name = 'COMBO'
    return type_name

ParameterType: TypeAlias = Literal['required', 'optional', 'hidden']

class _NameCheckCls(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        '''Check if the object is AnnotatedParam.'''
        type_name = get_cls_name(__instance)
        this_cls_name = get_cls_name(self)
        return type_name == this_cls_name    # due to complicated import dependencies, this will be used instead of isinstance(obj, AnnotatedParam)

class _GetableFunc(metaclass=_NameCheckCls):
    '''A special class to allow original function to be called when using `[]` syntax.'''
    
    origin_func: Callable
    
    def __init__(self, origin_func: Callable):
        self.origin_func = origin_func
    
    def __getitem__(self, vals):
        '''when using `INT[0, 10, 2]` syntax, INT(0, 10, 2) will be called.'''
        if isinstance(vals, tuple):
            return self.origin_func(*vals)
        return self.origin_func(vals)
    
    def __call__(self, *args, **kwargs):
        return self.origin_func(*args, **kwargs)

class AnnotatedParam(metaclass=_NameCheckCls):
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
    param_type: Optional[ParameterType] = None
    '''The input type of the param. `None` is available for return types.'''
   
    value_formatter: Optional[Callable[[Any], Any]] = None
    '''
    The convertor for changing the input type to the real value. E.g. Tensor(input) -> LATENT(before calling the node)
    It is not necessary to specify this.
    '''
    inner_type: Optional[Union[type, str]] = None
    '''The inner type of this param type. This is for special types, e.g. Array'''
    
    def __init__(self,
                 origin: Optional[Union[type, str, TypeAlias, Parameter, 'AnnotatedParam']],
                 extra_params: Optional[dict] = None,
                 tags: Optional[Union[List[str], Tuple[str, ...], Set[str]]] = None,
                 default: Optional[Any] = Parameter.empty,
                 param_name: Optional[str] = None,
                 param_type: Optional[ParameterType] = None,
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
        if param_type is not None and not valueTypeCheck(param_type, ParameterType):   # type: ignore
            raise TypeError(f'Unexpected type for param_type: {param_type}. It should be one of {ParameterType}.')
        
        if isinstance(origin, _GetableFunc):
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
                origin = Any
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
            
            origin = origin.origin_type
            
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
            origin = DynamicLiteral(*origin)  # turn into COMBO
        
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
                self.value_formatter = lambda val: _enum_convertor(self.origin_type, val)
            else:
                self.value_formatter = value_formatter
                if not self.value_formatter and hasattr(self.origin_type, '__ComfyValueFormatter__'):
                    formatter = self.origin_type.__dict__['__ComfyValueFormatter__']
                    if formatter is not None:
                        if not isinstance(formatter, (classmethod, staticmethod)):
                            raise TypeError(f'__ComfyValueFormatter__ should be a class or static method, but got {formatter}.')
                        self.value_formatter = formatter
        else:
            self.value_formatter = value_formatter
        
        self.param_type = param_type
        if not self.param_type and param:
            self.param_type = _get_input_param_type(param, self.tags)

    @property
    def _comfy_type(self)->Union[str, list]:
        '''
        Return the comfy type name(or list of value for COMBO type) for this param.
        For internal use only.
        '''
<<<<<<< HEAD
        if self.comfy_name:
            return self.comfy_name
        return _get_comfy_type_definition(self.origin_type, self.inner_type) # type: ignore
=======
        if self.comfy_type_name:
            return self.comfy_type_name
        
        if not self.origin_type:
            raise ValueError('The comfy_type_name is not specified and the origin_type is not specified either.')
        return _get_comfy_type_definition(self.origin_type)
>>>>>>> parent of ca58cd6 (Implement RGBA2RGB, RGBAThreshold)

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
            if not valueTypeCheck(val, self.origin_type):
                need_convert = True
        
        if need_convert:
            return self.value_formatter(val)
        return val

class InferenceContext:
    
    @class_or_ins_property  # type: ignore
    def _executor(cls_or_ins)-> 'PromptExecutor':
        '''The prompt executor instance.'''
        from comfyUI.execution import PromptExecutor
        return PromptExecutor() # since the PromptExecutor is a singleton, it is safe to create a new instance here
    
    prompt: 'PROMPT'
    '''The prompt of current execution. It is a dict containing all nodes' input for execution.'''
    
    extra_data: dict
    '''Extra data for the current execution.'''
    
    current_node: Optional['ComfyUINode'] = None
    '''The current node for this execution.'''
    
    def __init__(self, current_prompt: 'PROMPT', extra_data: dict):
        self.prompt = current_prompt
        self.extra_data = extra_data
       
    @property
    def node_pool(self)-> 'NodePool':
        '''
        Pool of all created nodes
        {node_id: node_instance, ...}
        '''
        return self._executor.node_pool
    
    @property
    def old_prompt(self)-> 'PROMPT':
        '''prompt from last execution'''    
        return self._executor.old_prompt
    
    @property
    def outputs_ui(self)-> 'NodeOutputs_UI':
        '''outputs for UI'''
        return self._executor.outputs_ui
    
    @property
    def outputs(self)-> 'NodeOutputs':
        '''current outputs in this execution'''
        return self._executor.outputs


_CT = TypeVar('_CT', bound='ComfyType')

class ComfyType(Protocol, Generic[_CT]):
    '''Protocol for hinting the special properties u can define in customized type for comfyUI.'''
    
    @classmethod
    def __ComfyValueFormatter__(cls, value)->Any:
        '''Convert the value to the real value for the type.'''
    
    def __ComfyDump__(self)->Any:
        '''You can define how this value should be dump to json when returning the response.'''
    
    @classmethod
    def __ComfyLoad__(cls: Type[_CT], data: Any)->_CT:
        '''You can define how to load the data from json to the object.'''
    
    __ComfyName__: ClassVar[str]
    '''If you have set this field, it would be the comfy name for this type.'''

class HIDDEN(ABC):
    '''
    The base class for all special hidden types. Class inherited from this class will must be treated as hidden type in node system.
    
    You could also define hidden types by naming params with prefix `_`.
    But all hidden types have special meanings/ special behaviors in ComfyUI,
    so it is recommended to use this class to define hidden types.
    '''
    def __class_getitem__(cls, tp: type):
        return Annotated[tp, AnnotatedParam(origin=tp, tags=['hidden'])]
    
    @abstractclassmethod
    def GetValue(cls: Type[_HT], context: InferenceContext)->Optional[_HT]:   # type: ignore
        '''All hidden types should implement this method to get the real value from current inference context.'''
        raise NotImplementedError

__all__ = ['get_comfy_name', 'ParameterType', 'AnnotatedParam', 'InferenceContext', 'ComfyType', 'HIDDEN']


# region basic types
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
globals()['INT'] = _GetableFunc(INT)    # trick for faking IDE to believe INT is a function

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
globals()['FLOAT'] = _GetableFunc(FLOAT)    # trick for faking IDE to believe FLOAT is a function

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
globals()['STRING'] = _GetableFunc(STRING)    # trick for faking IDE to believe STRING is a function

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
globals()['BOOLEAN'] = _GetableFunc(BOOLEAN)    # trick for faking IDE to believe BOOLEAN is a function

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
globals()['PATH'] = _GetableFunc(PATH)    # trick for faking IDE to believe PATH is a function

__all__.extend(['INT', 'FLOAT', 'STRING', 'BOOLEAN', 'PATH', ])
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

MODEL: TypeAlias = Annotated[BaseModel, AnnotatedParam(origin=BaseModel, comfy_name='MODEL')]
'''type hint for ComfyUI's built-in type `MODEL`.'''

IMAGE: TypeAlias = Annotated[Tensor, AnnotatedParam(origin=Tensor, comfy_name='IMAGE')]
'''Type hint for ComfyUI's built-in type `IMAGE`.
When multiple imgs are given, they will be concatenated as 1 tensor by ComfyUI.'''

MASK: TypeAlias = Annotated[Tensor, AnnotatedParam(origin=Tensor, comfy_name='MASK')]
'''Type hint for ComfyUI's built-in type `MASK`.
Similar to `IMAGE`, when multiple masks are given, they will be concatenated as 1 tensor by ComfyUI.'''

__all__.extend(['LATENT', 'VAE', 'CLIP', 'CONTROL_NET', 'MODEL', 'IMAGE', 'MASK'])
# endregion

# region comfyUI built-in options
def DynamicLiteral(*args: Union[str, int, float, bool])->TypeAlias:
    '''Literal annotation for dynamic options.'''
    literal_args = tuple(args)
    for option in literal_args:
        if not isinstance(option, (str, int, float, bool)):
            raise TypeError(f'Unexpected type: {option}. It should be one of str, int, float, bool.')
    t = Literal[""]
    t.__args__ = literal_args  # type: ignore
    return t
globals()['DynamicLiteral'] = _GetableFunc(DynamicLiteral)    # trick for faking IDE to believe DynamicLiteral is a function

COMFY_SCHEDULERS = DynamicLiteral(*KSampler.SCHEDULERS)
'''Literal annotation for choosing comfyUI's built-in schedulers.'''

COMFY_SAMPLERS = DynamicLiteral(*KSampler.SAMPLERS)
'''Literal annotation for choosing comfyUI's built-in samplers.'''

__all__.extend(['DynamicLiteral', 'COMFY_SCHEDULERS', 'COMFY_SAMPLERS',])
# endregion


# region return types
def Named(tp: Union[type, TypeAlias, AnnotatedParam], return_name: str):
    '''Type hint for specifying a named return value.'''
    return AnnotatedParam(tp, param_name=return_name)
globals()['Named'] = _GetableFunc(Named)    # trick for faking IDE to believe ReturnType is a function

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
class Lazy(Generic[_T]):
    '''
    Mark the type as lazy, for lazy loading in ComfyUI.
    This could only be used in input parameters.
    '''
    
    def __class_getitem__(cls, tp: Type[_T]):
        '''Support the type hint syntax, e.g. Lazy[int]'''
        real_type = tp.origin_type if isinstance(tp, AnnotatedParam) else tp
        return Annotated[real_type, AnnotatedParam(tp, tags=['lazy'])]
    
    _value: _T = None    # type: ignore
    
    _from_node_id: str = None   # type: ignore
    _from_slot: str = None  # type: ignore
    
    @overload
    def __init__(self, from_node_id: str, from_slot: int):
        '''Create a lazy type from another node's output.'''
    @overload
    def __init__(self, value: _T):
        '''Create a lazy type from a real value.'''
    
    def __init__(self, *args):  # type: ignore
        if len(args) not in (1, 2):
            raise ValueError(f'Invalid arguments: {args}. It should be `value` or `(from_node_id, from_slot)`.')
        if len(args) == 1:
            self._value = args[0]
        else:
            self._from_node_id, self._from_slot = args
    
    def _get_value(self)->_T:   # type: ignore
        if self._value is None:
            if not self._from_node_id or not self._from_slot:
                raise ValueError(f'Invalid lazy type: {self}. It should be created from another node.')
            self._value = PromptExecutor.Instance._get_node_output(self._from_node_id, self._from_slot) # type: ignore
        return self._value
    
    @property
    def value(self)->_T:
        '''The real value of the lazy type. The value will only be resolved when it is accessed.'''
        return self._get_value()

class Array(list, Generic[_T]):
    '''
    Mark the type as array, to allow the param accept multiple values and pack as a list.
    This could only be used in input parameters.
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


# region ComfyNode
class _ComfyUINodeMeta(_ProtocolMeta):
    def __instancecheck__(cls, instance: Any) -> bool:
        if hasattr(instance, '__IS_COMFYUI_NODE__') and instance.__IS_COMFYUI_NODE__:
            return True
        return super().__instancecheck__(instance)

@runtime_checkable
class ComfyUINode(Protocol, metaclass=_ComfyUINodeMeta):
    '''
    The type hint for comfyUI nodes. It is not necessary to inherit from this class.
    I have done some tricks to make it available for `isinstance` check.
    
    Note: 
        For better node customization, you can use the `NodeBase` class from `comfyUI.stable_renderer`,
        which is will do automatic registration and type hinting for you. You just need to define the `__call__` method
        and put type annotation on input/output parameters.
    '''
    
    # region internal attributes
    __IS_COMFYUI_NODE__: ClassVar[bool]
    '''internal flag for identifying comfyUI nodes.'''
    
    __IS_ADVANCED_COMFYUI_NODE__: ClassVar[bool] = False
    '''if u create nodes by inheriting from `NodeBase`, this flag will be set to True.'''
    
    __ADVANCED_NODE_CLASS__: ClassVar[Optional[Type['NodeBase']]] = None
    '''If u create nodes by inheriting from `NodeBase`, this will be the real node class.'''
    
    __real_node_instance__: 'NodeBase'
    '''real node instance after creation.'''
    
    _ID: str
    '''The unique id of the node. This will be assigned in runtime.'''
    @property
    def ID(self)->str:
        '''
        The unique id of the node.
        This will be assigned in runtime.
        '''
        return self._ID
    # endregion
    
    FUNCTION: str
    '''the target function name of the node'''
    
    DESCRIPTION: str
    '''the description of the node'''
    CATEGORY: str
    '''the category of the node. For searching'''
    
    INPUT_IS_LIST: bool
    '''whether the input is a list'''
    OUTPUT_IS_LIST: Tuple[bool, ...]
    '''whether the output is a list'''
    LAZY_INPUTS: Tuple[str, ...]
    '''the lazy input params of the node'''
    
    OUTPUT_NODE: bool
    '''
    Means the node have values that shown on UI.
    Too strange, the name and the meaning are not matched...
    '''

    @classmethod
    def INPUT_TYPES(cls) -> 'NodeInputDict':  # type: ignore
        '''All nodes for comfyUI should have this class method to define the input types.'''
    
    RETURN_TYPES: Tuple[str, ...]
    '''All return type names of the node.'''
    RETURN_NAMES: Tuple[str, ...]
    '''Names for each return values'''
    
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs)->bool:    # type: ignore
        '''If you have defined this method, it will be called to validate the input values.'''    
    
    def IS_CHANGED(self, *args, **kwargs): 
        '''
        This method will be called when the input value is changed. 
        It should have the same signature as the function which you have defined to called.
        '''

__all__.extend(['ComfyUINode'])
# endregion

# region type aliases for comfyUI's origin codes
NodeInputDict: TypeAlias = Dict[Literal['required', 'optional', 'hidden'], Dict[str, Any]]
'''
{input_type: {param_name: param_info, ...}, ...}
The dictionary containing input params' information of a node.
'''

@singleton(cross_module_singleton=True)
class NodePool(Dict[Union[str, Tuple[str, Union[str, Type[ComfyUINode]]]], ComfyUINode]):
    '''
    {node_id: node_instance, ...}
    The global pool of all created nodes.
    
    This node pool is not exactly the same structure as the original ComfyUI's node pool,
    because the original design is not that good, it set the node type name as a part of the key,
    but it is not necessary to do that (nodes always have unique id).
    '''
    
    @class_property # type: ignore
    def Instance(cls)->'NodePool':
        '''Get the global instance of the node pool.'''
        return cls.__instance__  # type: ignore
    
    def __setitem__(self, __key: Union[str, Tuple[str, Union[str, Type[ComfyUINode]]]], __value: ComfyUINode) -> None:
        '''
        possible key:
            - node_id: the unique id of the node
            - (node_id, node_cls_name): the unique id and class name of the node
            - (node_id, node_cls): the unique id and class of the node
        '''
        if isinstance(__key, tuple):
            if not len(__key) == 2:
                raise ValueError(f'Invalid key: {__key}, it should be a tuple with 2 elements(node_id, node_cls_name or node_cls).' )
            key = __key[0]
            node_type_name = __key[1] if isinstance(__key[1], str) else __key[1].__qualname__
            if key in self and type(self[key]).__qualname__ != node_type_name:
                raise ValueError(f'The node id {key} is already used by another node type {type(self[key]).__qualname__}.')
        else:
            key = __key
            
        return super().__setitem__(key, __value)    
    
    def __getitem__(self, __key: Union[str, Tuple[str, Union[str, Type[ComfyUINode]]]]) -> ComfyUINode:
        '''
        possible key:
            - node_id: the unique id of the node
            - (node_id, node_cls_name): the unique id and class name of the node
            - (node_id, node_cls): the unique id and class of the node
            
        When a tuple is used as the key, it will create a new node if the node is not found.
        '''
        if isinstance(__key, tuple):
            if not len(__key) == 2:
                raise ValueError(f'Invalid key: {__key}, it should be a tuple with 2 elements(node_id, node_cls_name or node_cls).' )
            node_id = __key[0]
            node_type_name = __key[1] if isinstance(__key[1], str) else __key[1].__qualname__
            if node_id in self and type(self[node_id]).__qualname__ != node_type_name:
                raise ValueError(f'The node id {node_id} is already used by another node type {type(self[node_id]).__qualname__}.')
            else:
                node_type = __key[1]
                if isinstance(node_type, str):
                    from comfyUI.nodes import NODE_CLASS_MAPPINGS
                    node_type = NODE_CLASS_MAPPINGS[node_type]
                node = node_type()
                setattr(node, '_ID', node_id)
                # '__IS_COMFYUI_NODE__' should already be set in `nodes.py`
                self[node_id] = node
        else:
            node_id = __key
        
        return super().__getitem__(node_id)
    
    def get_node(self, node_id: Union[str, int], node_cls: Union[str, type])->ComfyUINode:
        '''
        Get the node instance by node_id and node_cls_name.
        If the node is not found, create a new one and return it.
        '''
        return self[(node_id, node_cls)]   # type: ignore

    class NodePoolKey(str):
        '''
        The key for node pool.
        This class is for making the new datatype `NodePool` compatible with all old code.
        '''
        
        node: ComfyUINode
        '''The node instance.'''
        
        def __new__(cls, node_id: str, node: ComfyUINode):
            return super().__new__(cls, node_id)
        
        def __init__(self, node_id: str, node: ComfyUINode):
            self.node = node
        
        def __getitem__(self, item: int)->str:
            if item not in (0, 1):
                raise IndexError(f'Invalid index: {item}, it should be 0 or 1.')
            if item == 0:
                return self # return the node id
            else:
                return type(self.node).__qualname__
        
        def __eq__(self, other: Union[str, Tuple[str, Union[str, Type[ComfyUINode]]]]) -> bool:
            if isinstance(other, str):
                return super().__eq__(other)
            elif isinstance(other, tuple) and len(other)>=1:
                return super().__eq__(other[0])
            else:
                return super().__eq__(other)
            
        def __hash__(self) -> int:
            return super().__hash__()   # not useless, it is necessary for dict key
        
        @property
        def node_type_name(self)->str:
            return self[1]
        
        @property
        def node_id(self)->str:
            return self
        
        @property
        def node_type(self)->Type[ComfyUINode]:
            from comfyUI.nodes import NODE_CLASS_MAPPINGS
            return NODE_CLASS_MAPPINGS[self.node_type_name]

    def __iter__(self) -> Iterator['NodePool.NodePoolKey']:
        '''Iterate through the node pool.'''
        for node_id, node in self.items():
            yield self.NodePoolKey(node_id, node)   # type: ignore

class NodeBindingParam(Tuple[str, int]):
    '''
    (node_id, output_slot_index)
    The tuple contains the information that the input value of a node is from another node's output.
    '''
    
    def __repr__(self):
        return f'NodeBindingParam({self[0]}, {self[1]})'
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Sequence) and len(__value) ==2:
            return self[0] == __value[0] and self[1] == __value[1]
        return super().__eq__(__value)
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    @property
    def from_node_id(self)->str:
        '''The node id of the source node. Wrapper for the first element of the tuple.'''
        return self[0]
    
    @property
    def from_node(self)->ComfyUINode: # type: ignore
        return NodePool.Instance[self.from_node_id]  # type: ignore
    
    @property
    def output_slot_index(self)->int:
        '''The output slot index of the source node. Wrapper for the second element of the tuple.'''
        return self[1]

    @staticmethod
    def InstanceCheck(value: Any)->TypeGuard['NodeBindingParam']:
        if not isinstance(value, type):
            value = type(value)
        type_name = get_cls_name(value)
        return type_name == 'NodeBindingParam'  # due to complicated importing & compatibility issues, we use name checking here.

class NodeInputs(Dict[str, Union[NodeBindingParam, Any]]):
    '''
    {param_name: input_value, ...}
    
    There are 2 types of dict value:
        - [node_id(str), output_slot_index(int)]: the input is from another node
        - Any: the input is a value
    '''
    
    def _should_convert_to_bind_type(self, value: list)->bool:
        if NodeBindingParam.InstanceCheck(value):
            return False    # already binding
        if len(value)!=2 or not isinstance(value[0], str) or not isinstance(value[1], int): 
            return False # not binding
        return True     # seems like a binding, e.g. ['1', 0] means from node_id=1 and output_slot_index=0
    
    def _format_values(self):
        for key, value in tuple(self.items()):
            if isinstance(value, list):
                if self._should_convert_to_bind_type(value):
                    self[key] = NodeBindingParam(value)
    
    
    node_id: Optional[str] = None
    '''The node id of the source node. It is only available when the input is from another node.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._format_values() # make sure all values are Any/FromNodeInput

    def __setitem__(self, __key: str, __value: Union[NodeBindingParam, Any]) -> None:
        if (isinstance(__value, list) and
            len(__value)==2 and 
            not isinstance(__value, NodeBindingParam) and
            isinstance(__value[0], str) and
            isinstance(__value[1], int)):
            
            __value = NodeBindingParam(*__value)
        super().__setitem__(__key, __value)

    @staticmethod
    def InstanceCheck(value: Any)->TypeGuard['NodeInputs']:
        if not isinstance(value, type):
            value = type(value)
        type_name = get_cls_name(value)
        return type_name == 'NodeInputs'

class NodeType(str):
    
    _real_cls: type = None
    
    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if isinstance(other, str):
            return super().__eq__(other)
        elif isinstance(other, ComfyUINode):
            return super().__eq__(type(other).__qualname__.split('.')[-1])
        return super().__eq__(other)
    
    @property
    def real_node_cls(self)->Type[ComfyUINode]:
        if not self._real_cls:
            from comfyUI.nodes import NODE_CLASS_MAPPINGS
            self._real_cls = NODE_CLASS_MAPPINGS[self]
        return self._real_cls
    
    @staticmethod
    def InstanceCheck(value: Any)->TypeGuard['NodeType']:
        if not isinstance(value, type):
            value = type(value)
        type_name = get_cls_name(value)
        return type_name == 'NodeType'

class PROMPT(Dict[str, Dict[Literal['inputs', 'class_type', 'is_changed'], Any]], HIDDEN):
    '''
    {node_id: info_dict}
    Info dict may contain:
        - inputs: the input data for the node
        - class_type: just the class name of the node(not type! bad name!)
        - is_changed: the value for identifying whether the input is changed(if u have defined `IS_CHANGED` method)

    Prompt is actually a dict containing all nodes' input for execution.
    '''
    
    def _format_prompt(self):
        '''Make sure all node inputs are set as value/NodeBindingParam'''
        for _, node_info_dict in self.items():
            if 'inputs' in node_info_dict:
                if not NodeInputs.InstanceCheck(node_info_dict['inputs']):
                    node_info_dict['inputs'] = NodeInputs(node_info_dict['inputs'])
            if 'class_type' in node_info_dict:
                if not NodeType.InstanceCheck(node_info_dict['class_type']):
                    node_info_dict['class_type'] = NodeType(node_info_dict['class_type']) 
            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._format_prompt() # tidy up types
    
    @classmethod
    def GetValue(cls, context: InferenceContext):
        '''Get the real value from current inference context.'''
        return context.prompt
    
    def get_node_inputs(self, node: Union[str, ComfyUINode])->NodeInputs:
        '''
        The input data for the node. The wrapper for `inputs` key.
        Each node in prompt dict must have a input data, so this property must be available.
        '''
        if not isinstance(node, str):
            node = node.ID
        return self[node]['inputs']
    
    def get_class_type(self, node: Union[str, ComfyUINode])->str:
        '''
        The class name of the node. The wrapper for `class_type` key.
        Each node in prompt dict must have a class name, so this property must be available.
        '''
        if not isinstance(node, str):
            node = node.ID
        return self[node]['class_type']   # type: ignore
    
    def get_class_def(self, node: Union[str, ComfyUINode])->Type[ComfyUINode]:
        '''return the class type of the node. The wrapper for `class_type` key.'''
        cls_type = self.get_class_type(node)
        from comfyUI.nodes import NODE_CLASS_MAPPINGS
        return NODE_CLASS_MAPPINGS[cls_type]
    
    def get_is_changed_validation_val(self, node: Union[str, ComfyUINode])->Optional[Any]:
        '''
        The value for identifying whether the input is changed(if u have defined `IS_CHANGED` method).
        Wrapper for `is_changed` key.
        
        This method returns optional value, because the `IS_CHANGED` method is not necessary to be defined in comfyUI nodes.
        '''
        if not isinstance(node, str):
            node = node.ID
        return self[node].get('is_changed', None)

class EXTRA_PNG_INFO(Dict[str, Any], HIDDEN):
    '''Extra information for saving png file.'''
    
    __ComfyName__: ClassVar[str] = 'EXTRA_PNGINFO'
    
    @classmethod
    def GetValue(cls: Type[_HT], context: InferenceContext)->Optional[_HT]:   # type: ignore
        '''All hidden types should implement this method to get the real value from current inference context.'''
        if 'extra_pnginfo' in context.extra_data:
            return context.extra_data['extra_pnginfo']
        return None

@deprecated(reason='unique id of node can be gotten by node.ID directly. It is not necessary to use this type anymore.')
class UNIQUE_ID(str, HIDDEN):
    '''The unique ID of current running node.'''
    
    @classmethod
    def GetValue(cls: Type[_HT], context: InferenceContext)->Optional[_HT]:
        cur_node = context.current_node
        if cur_node is not None:
            return cur_node.ID  # type: ignore
        return None

StatusMsg: TypeAlias = List[Tuple[str, Dict[str, Any]]]
'''
The status message type for PromptExecutor.
[(event, {msg_key: msg_value, ...}), ...]
'''

NodeOutputs_UI: TypeAlias = Dict[str, Dict[str, Any]]
'''All outputs of nodes for ui'''
NodeOutputs: TypeAlias = Dict[str, List[Any]]
'''
All outputs of nodes for execution.
{node id: [output1, output2, ...], ...}
'''

QueueTask: TypeAlias = Tuple[Union[int, float], str, PROMPT, dict, list]
'''
The type hint for the queue task in execution.PromptQueue's inner items
Items:
    - number (int? float? seems int but i also saw number=float(...), too strange, and idk wtf is this for)
    - prompt_id (str, random id by uuid4)
    - prompt    (for PromptExecutor.execute method)
    - extra_data
    - outputs_to_execute
'''


__all__.extend(['NodeInputDict', 'NodePool', 'NodeBindingParam', 'NodeInputs', 
                'PROMPT', 'EXTRA_PNG_INFO', 'UNIQUE_ID', 
                'StatusMsg', 'NodeOutputs_UI', 'NodeOutputs', 'QueueTask'])
# endregion

