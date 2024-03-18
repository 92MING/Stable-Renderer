from collections.abc import Iterator
from torch import Tensor
from typing import (Union, TypeAlias, Annotated, Literal, Optional, List, get_origin, get_args, ForwardRef, 
                    Any, TypeVar, Generic, Type, overload, Protocol, Dict, Tuple)
from inspect import Parameter
from enum import Enum

from common_utils.decorators import singleton, class_property
from comfy.samplers import KSampler
from comfy.sd import VAE as comfy_VAE, CLIP as comfy_CLIP
from comfy.controlnet import ControlBase as comfy_ControlNetBase
from comfy.model_base import BaseModel


_SPECIAL_TYPE_NAMES = {
    str: "STRING",
    bool: "BOOLEAN",
    int: "INT",
    float: "FLOAT",
}

def _get_comfy_type_definition(tp):
    '''
    return the type name or list of value for COMBO type.
    e.g. int->"INT", Literal[1, 2, 3]->[1, 2, 3]
    '''
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
                    return arg._comfy_type_definition
            raise TypeError(f'Unexpected Annotated type: {tp}')
        else:
            raise TypeError(f'Unexpected type annotation: {tp}')

    if isinstance(tp, str): # string annotation
        return tp.upper()     # default return type name to upper case
    
    if isinstance(tp, ForwardRef):
        return tp.__forward_arg__.upper()
    
    if tp == Any or tp == Parameter.empty:
        return "*"  # '*' represents any type in litegraph
    
    if hasattr(tp, '__qualname__'):
        return tp.__qualname__.split('.')[-1].upper()
    elif hasattr(tp, '__name__'):
        return tp.__name__.split('.')[-1].upper()
    else:
        raise TypeError(f'Cannot get type name for {tp}')

class AnnotatedParam:
    '''Annotation of parameters for ComfyUI's node system.'''
    
    origin_type: Optional[Union[type, TypeAlias]] = None
    '''
    The origin type of the parameter.
    It is not necessary to specify this, but if `comfy_type_name` is not specified, this is required.
    '''
    extra_params: Optional[dict] = None
    '''
    The extra parameters for the type, e.g. min, max, step for INT type, etc.
    This is usually for building special param setting in ComfyUI.
    '''
    tags: Optional[List[str]] = None
    '''special tags, for labeling special params, e.g. lazy, ui, reduce, ...'''
    default: Optional[Any] = None
    '''The default value for the parameter.'''
    comfy_type_name: Optional[str] = None
    '''The name for registering the type in ComfyUI's node system, e.g. INT for int, FLOAT for float, etc.'''
    
    def __init__(self,
                origin_type: Optional[Union[type, TypeAlias, 'AnnotatedParam']] = None,
                extra_params: Optional[dict] = None,
                tags: Optional[List[str]] = None,
                default: Optional[Any] = None,
                comfy_type_name: Optional[str] = None):
        '''
        Args:
            - origin_type: The origin type of the parameter. If you are using AnnotatedParam as type hint, 
                           values will be inherited from the origin type, and all other parameters will be used to update/extending the origin type.
            - extra_params: The extra parameters for the type, e.g. min, max, step for int type, etc.
            - tags: special tags, for labeling special params, e.g. lazy, ui, reduce, ...
            - default: The default value for the parameter.
            - comfy_type_name: The name for registering the type in ComfyUI's node system, e.g. INT for int, FLOAT for float, etc.
        '''
        if not origin_type and not comfy_type_name:
            raise ValueError('Either origin_type or comfy_type_name should be specified.')
        
        if isinstance(origin_type, AnnotatedParam):
            param = origin_type
            
            origin_type = origin_type.origin_type
            
            extra_params = extra_params or {}
            extra_params.update(param.extra_params or {})
            if not extra_params:
                extra_params = None
            
            tags = tags or []
            tags.extend(param.tags or [])
            if tags:
                tags = list(set(tags))
            else:
                tags = None
                
            default = default if default is not None else param.default
            comfy_type_name = comfy_type_name or param.comfy_type_name
            
        self.origin_type = origin_type
        self.extra_params = extra_params
        self.tags = tags
        self.default = default
        self.comfy_type_name = comfy_type_name

    @property
    def _comfy_type_definition(self)->Union[str, list]:
        '''
        Return the comfy type name(or list of value for COMBO type) for this param.
        For internal use only.
        '''
        if self.comfy_type_name:
            return self.comfy_type_name
        
        if not self.origin_type:
            raise ValueError('The comfy_type_name is not specified and the origin_type is not specified either.')
        return _get_comfy_type_definition(self.origin_type)

    @property
    def _clone(self)->'AnnotatedParam':
        '''Clone this instance.'''
        return AnnotatedParam(origin_type=self.origin_type, 
                              extra_params=self.extra_params, 
                              tags=self.tags, 
                              default=self.default, 
                              comfy_type_name=self.comfy_type_name)

    @property
    def _dict(self)->dict:
        '''Return the dict representation of this instance'''
        data = self.extra_params or {}
        if self.default:
            data['default'] = self.default
        return data
    
    @property
    def _tuple(self)->tuple:
        '''return the tuple form, e.g. ("INT", {...})'''
        data_dict = self._dict
        if data_dict:
            return (self._comfy_type_definition, data_dict)
        else:
            return (self._comfy_type_definition, )

__all__ = ['AnnotatedParam', ]


# region common types
def INT(min: int=0, max: int = 0xffffffffffffffff, step:int=1, display: Literal['number', 'slider', 'color']='number')->Annotated:
    '''
    This is the int type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: INT(0, 10, 2)=0): ...
    
    You can still use `int` for type annotation.
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider', 'color'):
        raise ValueError(f'Invalid value for display: {display}. It should be one of "number", "slider", "color".')
    data = {"min": min, "max": max, "step": step, "display": display}
    return Annotated[int, AnnotatedParam(origin_type=int, extra_params=data, comfy_type_name='INT')]
    
def FLOAT(min=0, max=100, step=0.01, round=0.001, display: Literal['number', 'slider']='number')->Annotated:
    '''
    This is the float type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: FLOAT(0, 10, 2)=0): ...
    
    You can still use `float` for type annotation.
    '''
    display = display.lower()   # type: ignore
    if display not in ('number', 'slider'):
        raise ValueError(f'Invalid value for display: {display}, It should be one of "number", "slider".')
    data = {"min": min, "max": max, "step": step, "round": round, "display": display}
    return Annotated[float, AnnotatedParam(origin_type=float, extra_params=data, comfy_type_name='FLOAT')]

def STRING(multiline=False, forceInput: bool=False)->Annotated:
    '''
    This is the str type annotation for containing more information. 
    For default values, set by `=...`, e.g. def f(x: STRING(multiline=True)=""): ...
    
    You can still use `str` for type annotation.
    '''
    data = {"multiline": multiline, "forceInput": forceInput}
    return Annotated[str, AnnotatedParam(origin_type=str, extra_params=data, comfy_type_name='STRING')]

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

CLIP: TypeAlias = Annotated[comfy_CLIP, AnnotatedParam(origin_type=comfy_CLIP, comfy_type_name='CLIP')]
'''type hint for ComfyUI's built-in type `CLIP`.'''

VAE: TypeAlias = Annotated[comfy_VAE, AnnotatedParam(origin_type=comfy_VAE, comfy_type_name='VAE')]
'''type hint for ComfyUI's built-in type `VAE`.'''

CONTROL_NET: TypeAlias = Annotated[comfy_ControlNetBase, AnnotatedParam(origin_type=comfy_ControlNetBase, comfy_type_name='CONTROL_NET')]
'''
Type hint for ComfyUI's built-in type `CONTROL_NET`.
This type includes all control networks in ComfyUI, including `ControlNet`, `T2IAdapter`, and `ControlLora`.
'''

MODEL: TypeAlias = Annotated[BaseModel, AnnotatedParam(origin_type=BaseModel, comfy_type_name='MODEL')]
'''type hint for ComfyUI's built-in type `MODEL`.'''

IMAGE: TypeAlias = Annotated[Tensor, AnnotatedParam(origin_type=Tensor, comfy_type_name='IMAGE')]
'''Type hint for ComfyUI's built-in type `IMAGE`.
When multiple imgs are given, they will be concatenated as 1 tensor by ComfyUI.'''

MASK: TypeAlias = Annotated[Tensor, AnnotatedParam(origin_type=Tensor, comfy_type_name='MASK')]
'''Type hint for ComfyUI's built-in type `MASK`.
Similar to `IMAGE`, when multiple masks are given, they will be concatenated as 1 tensor by ComfyUI.'''

__all__.extend(['INT', 'FLOAT', 'STRING', 'LATENT', 'VAE', 'CLIP', 'CONTROL_NET', 'MODEL', 'IMAGE', 'MASK'])
# endregion

# region comfyUI built-in options
COMFY_SCHEDULERS: TypeAlias = Literal[""]
'''Literal annotation for choosing comfyUI's built-in schedulers.'''
COMFY_SCHEDULERS.__args__ = tuple(KSampler.SCHEDULERS)  # type: ignore

COMFY_SAMPLERS: TypeAlias = Literal[""]
'''Literal annotation for choosing comfyUI's built-in samplers.'''
COMFY_SAMPLERS.__args__ = tuple(KSampler.SAMPLERS)  # type: ignore

__all__.extend(['COMFY_SCHEDULERS', 'COMFY_SAMPLERS',])
# endregion

# region special types for comfyUI
def UI(tp: Union[type, TypeAlias, AnnotatedParam])->AnnotatedParam:
    '''
    To define a output type that should be shown in UI.
    Obviously not all types are supported for UI, but I can't find a document for that.
    '''
    if isinstance(tp, AnnotatedParam):
        if tp.tags is None:
            tp.tags = []
        tp.tags.append('ui')
    else:
        tp = AnnotatedParam(tp, tags=['ui'])
    return tp

UI_IMAGE: TypeAlias = Annotated['IMAGE', AnnotatedParam(comfy_type_name='IMAGE', extra_params={'animated': False}, tags=['ui'])]
'''Specify the return image should be shown in UI.'''

UI_ANIMATION: TypeAlias = Annotated['IMAGE', AnnotatedParam(comfy_type_name='IMAGE', tags=['ui', 'animated'])]
'''Specify the return animated image should be shown in UI.'''

UI_LATENT: TypeAlias = Annotated[LATENT, AnnotatedParam(comfy_type_name='LATENT', tags=['ui'])]
'''Specify the return latent should be shown in UI.'''

__all__.extend(['UI', 'UI_IMAGE', 'UI_ANIMATION', 'UI_LATENT'])

_T = TypeVar('_T')

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
    def __init__(self, from_node_id: str, from_slot: str):
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
        if self._value is not None:
            return self._value
        else:
            pool: NodePool = NodePool.Instance  # type: ignore
            node = pool[self._from_node_id]
            
            
    @property
    def value(self)->_T:
        '''The real value of the lazy type. The value will only be resolved when it is accessed.'''
        return self._get_value()

def Reduce(tp: Union[type, TypeAlias, AnnotatedParam]) -> Annotated:
    '''
    Mark the type as reduce, to allow the param accept multiple values and pack as a list.
    This could only be used in input parameters.
    '''
    real_type = tp.origin_type if isinstance(tp, AnnotatedParam) else tp
    return Annotated[real_type, AnnotatedParam(tp, tags=['reduce'])]


__all__.extend(['Lazy', 'Reduce'])
# endregion


class ComfyUINode(Protocol):
    '''
    The type hint for comfyUI nodes. It is not necessary to inherit from this class.
    
    Note: 
        For better node customization, you can use the `NodeBase` class from `comfyUI.stable_renderer`,
        which is will do automatic registration and type hinting for you. You just need to define the `__call__` method
        and put type annotation on input/output parameters.
    '''
    
    _ID: str
    '''The unique id of the node. This will be assigned in runtime.'''
    @property
    def ID(self)->str:
        '''
        The unique id of the node.
        This will be assigned in runtime.
        '''
        return self._ID

    
    FUNCTION: str
    '''the target function name of the node'''
    
    DESCRIPTION: str
    '''the description of the node'''
    
    INPUT_IS_LIST: bool
    '''whether the input is a list'''
    OUTPUT_IS_LIST: Tuple[bool, ...]
    '''whether the output is a list'''
    
    LAZY_INPUTS: Tuple[str, ...]
    '''the input names which are lazy'''
    REDUCE_INPUTS: Tuple[str, ...]
    '''the input names which are reduce'''

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
    def VALIDATE_INPUTS(cls, *args, **kwargs)->bool:
        '''If you have defined this method, it will be called to validate the input values.'''    
    
    def IS_CHANGED(self, *args, **kwargs): 
        '''
        This method will be called when the input value is changed. 
        It should have the same signature as the function which you have defined to called.
        '''

__all__.extend(['ComfyUINode'])


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
            key = __key[0]
            node_type_name = __key[1] if isinstance(__key[1], str) else __key[1].__qualname__
            if key in self and type(self[key]).__qualname__ != node_type_name:
                raise ValueError(f'The node id {key} is already used by another node type {type(self[key]).__qualname__}.')
            else:
                node_type = __key[1]
                if isinstance(node_type, str):
                    from comfyUI.nodes import NODE_CLASS_MAPPINGS
                    node_type = NODE_CLASS_MAPPINGS[node_type]
                node = node_type()
                self[key] = node
        else:
            key = __key
        
        return super().__getitem__(key)
    
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
                return self
            else:
                return type(self.node).__qualname__
            
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

class FromNodeInput(Tuple[str, str]):
    '''
    (node_id, output_slot_index)
    The tuple contains the information that the input value of a node is from another node's output.
    '''
    
    @property
    def from_node_id(self)->str:
        '''The node id of the source node. Wrapper for the first element of the tuple.'''
        return self[0]
    
    @property
    def output_slot_index(self)->str:
        '''The output slot index of the source node. Wrapper for the second element of the tuple.'''
        return self[1]
    
    @property
    def from_node(self)->ComfyUINode:
        pool = NodePool()   # node pool is a singleton, here means getting the global node pool
        return pool[self.from_node_id]
    

class NodeInputs(Dict[str, Union[FromNodeInput, Any]]):
    '''
    {param_name: input_value, ...}
    
    There are 2 types of dict value:
        - [node_id, output_slot_index]: the input is from another node
        - Any: the input is a value
    '''
    pass
    

class Prompt(Dict[str, Dict[Literal['inputs', 'class_type', 'is_changed'], Any]]):
    '''
    {node_id: info_dict}
    Info dict may contain:
        - inputs: the input data for the node
        - class_type: just the class name of the node(not type! bad name!)
        - is_changed: the value for identifying whether the input is changed(if u have defined `IS_CHANGED` method)

    Prompt is actually a dict containing all nodes' input for execution.
    '''
    
    @property
    def inputs(self)->NodeInputs:
        '''
        The input data for the node. The wrapper for `inputs` key.
        Each node in prompt dict must have a input data, so this property must be available.
        '''
        return self['inputs']   # type: ignore
    
    @property
    def class_type(self)->str:
        '''
        The class name of the node. The wrapper for `class_type` key.
        Each node in prompt dict must have a class name, so this property must be available.
        '''
        return self['class_type']   # type: ignore
    
    @property
    def class_def(self)->Type[ComfyUINode]:
        '''return the class type of the node. The wrapper for `class_type` key.'''
        cls_type = self.class_type
        from comfyUI.nodes import NODE_CLASS_MAPPINGS
        return NODE_CLASS_MAPPINGS[cls_type]
    
    @property
    def is_changed_validation_val(self)->Optional[Any]:
        '''
        The value for identifying whether the input is changed(if u have defined `IS_CHANGED` method).
        Wrapper for `is_changed` key.
        '''
        return self.get('is_changed', None)

StatusMsg: TypeAlias = List[Tuple[str, Dict[str, Any]]]
'''
The status message type for PromptExecutor.
[(event, {msg_key: msg_value, ...}), ...]
'''

NodeOutputs_UI: TypeAlias = Dict[str, Dict[str, Any]]
'''All outputs of nodes for ui'''
NodeOutputs: TypeAlias = Dict[str, List[Any]]
'''All outputs of nodes for execution'''

QueueTask: TypeAlias = Tuple[Union[int, float], str, Prompt, dict, list]
'''
The type hint for the queue task in execution.PromptQueue's inner items
Items:
    - number (int? float? seems int but i also saw number=float(...), too strange, and idk wtf is this for)
    - prompt_id (str, random id by uuid4)
    - prompt    (for PromptExecutor.execute method)
    - extra_data
    - outputs_to_execute
'''


__all__.extend(['NodeInputDict', 
                
                'NodePool', 'FromNodeInput', 'NodeInputs', 'Prompt', 
                
                'StatusMsg', 
                
                'NodeOutputs_UI', 'NodeOutputs', 'QueueTask'])
# endregion

