import torch
from deprecated import deprecated
from typing import (Union, TypeAlias, Literal, Optional, List, Any, Type, Dict, Tuple, TYPE_CHECKING, 
                    Sequence, TypeVar, Generic, Set, NamedTuple, Union, Callable, Any)
from attr import attrs, attrib
from dataclasses import dataclass
from common_utils.debug_utils import ComfyUILogger
from common_utils.type_utils import get_cls_name, NameCheckMetaCls, valueTypeCheck
from common_utils.decorators import singleton, class_property, class_or_ins_property, Overload

from ._utils import get_node_type_by_name

if TYPE_CHECKING:
    from .node_base import ComfyUINode
    from .hidden import PROMPT, FrameData, BakingData
    from comfyUI.execution import PromptExecutor
    from comfyUI.adapters import Adapter

@attrs
class InferenceContext:
    
    @class_or_ins_property  # type: ignore
    def _executor(cls_or_ins)-> 'PromptExecutor':
        '''The prompt executor instance.'''
        from comfyUI.execution import PromptExecutor
        return PromptExecutor() # since the PromptExecutor is a singleton, it is safe to create a new instance here
    
    prompt:"PROMPT" = attrib()
    '''The prompt of current execution. It is a dict containing all nodes' input for execution.'''
    
    extra_data: dict = attrib(factory=dict)
    '''Extra data for the current execution.'''
    
    _current_node_id: Optional[str] = attrib(default=None, alias='current_node_id')
    '''The current node for this execution.'''
    
    old_prompt: Optional["PROMPT"] = attrib(default=None)
    '''The prompt from last execution.'''
    
    outputs:'NodeOutputs' = attrib(factory=dict)
    '''The outputs of the nodes for execution. {node id: [output1, output2, ...], ...}'''
    
    outputs_ui: 'NodeOutputs_UI' = attrib(factory=dict)
    '''The outputs of the nodes for ui. {node id: {output_name: output_value, ...}, ...}'''
    
    executed_node_ids: Set[str] = attrib(factory=set)
    '''The executed nodes' ids.'''
    
    frame_data: Optional["FrameData"] = attrib(default=None)
    '''The frame data for current execution.'''
    
    baking_data: Optional["BakingData"] = attrib(default=None)
    '''The baking data for current execution.'''
    
    status_messages: 'StatusMsgs' = attrib(factory=list)
    '''The status messages for current execution. [(event, {msg_key: msg_value, ...}), ...]'''
    
    success:bool = attrib(default=False)
    '''Whether the execution is successful.'''
    
    @property
    def prompt_id(self)->str:
        '''The prompt id for current execution.'''
        return self.prompt.id

    @property
    def current_node_id(self)->Optional[str]:
        return self._current_node_id
    
    @current_node_id.setter
    def current_node_id(self, value: Union[str, int, "ComfyUINode"]):
        from .node_base import ComfyUINode
        if isinstance(value, int):
            value = str(value)
        elif isinstance(value, ComfyUINode):
            value = value.ID
        self._current_node_id = value
    
    @property
    def current_node_type(self)->Optional[str]:
        if not self.current_node_id:
            return None
        return self.prompt[self.current_node_id]['class_type']
    
    @property
    def current_node(self)->Optional["ComfyUINode"]:
        if not self.current_node_id:
            return None
        return NodePool().get(self.current_node_id, self.current_node_type, create_new=True)   # type: ignore

    @current_node.setter
    def current_node(self, value: Union[str, int, "ComfyUINode"]):
        self.current_node_id = value

    def destroy(self):
        if self.old_prompt:
            self.old_prompt.destroy()   # current prompt no need to be destroyed, it will be destroyed by the context

@singleton(cross_module_singleton=True)
class NodePool(Dict[Tuple[str, str], "ComfyUINode"]):
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
    
    def __setitem__(self, __key: Union[int, str, Tuple[str, Union[str, Type["ComfyUINode"]]]], __value: "ComfyUINode") -> None:
        '''
        possible key:
            - node_id: the unique id of the node
            - (node_id, node_cls_name): the unique id and class name of the node
            - (node_id, node_cls): the unique id and class of the node
        '''
        if isinstance(__key, tuple):
            if not len(__key) in (1, 2):
                raise ValueError(f'Invalid key: {__key}, it should be a tuple with 2 elements(node_id, node_cls_name or node_cls).' )
            key = str(__key[0]) if not isinstance(__key[0], str) else __key[0]
            node_type_name = None if len(__key)==1 else __key[1]
        
            if node_type_name:
                if isinstance(node_type_name, type):
                    node_type_name = get_cls_name(node_type_name) 
                if node_type_name != get_cls_name(__value):
                    raise ValueError(f'The given node type `{node_type_name}` is not same as the node instance type `{get_cls_name(__value)}`.')
                if (key, node_type_name) in self and type((key, node_type_name)).__qualname__ != node_type_name:
                    raise ValueError(f'The node id {key} is already used by another node type {type(self[key]).__qualname__}.')
        else:
            key = str(__key) if isinstance(__key, int) else __key
            node_type_name = None
        
        if not node_type_name:
            node_type_name = get_cls_name(__value)
            
        return super().__setitem__((key, node_type_name), __value)    
    
    def __getitem__(self, __key: Union[int, str, Tuple[str, Union[str, Type["ComfyUINode"]]]]) -> "ComfyUINode":
        '''
        possible key:
            - node_id: the unique id of the node
            - (node_id, node_cls_name): the unique id and class name of the node
            - (node_id, node_cls): the unique id and class of the node
            
        When a tuple is used as the key, it will create a new node if the node is not found.
        '''
        if isinstance(__key, tuple):
            if not len(__key) in (1, 2):
                raise ValueError(f'Invalid key: {__key}, it should be a tuple with 2 elements(node_id, node_cls_name or node_cls).' )
            if len(__key) == 2:
                node_id = str(__key[0]) if isinstance(__key[0], int) else __key[0]
                node_type_name = __key[1] if isinstance(__key[1], str) else __key[1].__qualname__
            else:
                node_id = str(__key) if isinstance(__key, int) else __key
                node_type_name = None
        else:
            node_id = str(__key) if isinstance(__key, int) else __key
            node_type_name = None
        
        if node_type_name and not isinstance(node_type_name, str):
            node_type_name = str(node_type_name)
        if (node_id, node_type_name) in self and type(self[node_id]).__qualname__ != node_type_name:
            raise KeyError(f'The node id {node_id} is already used by another node type {type(self[node_id]).__qualname__}.')
        
        if node_type_name is None:
            for key in tuple(super().keys()):
                if key[0] == node_id:
                    return super().__getitem__(key)
            raise KeyError(f'Node not found: {node_id}. If you wanna create a new node, please provide the node type name.')
        
        elif (node_id, node_type_name) not in self:
            node_type = get_node_type_by_name(node_type_name)
            if not node_type:
                raise KeyError(f'Invalid node type name: {node_type_name}.')
            node = node_type()
            setattr(node, 'ID', node_id)
            # '__IS_COMFYUI_NODE__' should already be set in `nodes.py`
            super().__setitem__((node_id, node_type_name), node)
        
        node = super().__getitem__((node_id, node_type_name))
        setattr(node, 'ID', node_id) if not hasattr(node, 'ID') else None
        return node
    
    @Overload
    def get(self, node_id: Union[str, int], node_type: Union[str, Type["ComfyUINode"]]=None, create_new=True, raise_err=False):  # type: ignore
        '''Get the node instance by node_id.'''
        node_id = str(node_id) if isinstance(node_id, int) else node_id
        node_type = get_cls_name(node_type) if (node_type is not None and isinstance(node_type, type)) else node_type
        
        if not create_new:
            node = self[node_id]    # will raise KeyError if not found
            if node_type and node_type != get_cls_name(node):
                if raise_err:
                    raise ValueError(f'The given node type `{node_type}` is not same as the node instance type `{get_cls_name(node)}`.')
                return None
            return node
        else:
            try:
                if not node_type:
                    node = self[node_id]    # will raise KeyError if not found
                else:
                    print("Node id", node_id, " node type ", node_type)
                    return self[(node_id, node_type)]   # will create if not found
            except KeyError as e:
                if raise_err:
                    raise e
                return None
                
    @Overload
    def get(self, key: Tuple[Union[str, int], str], default=None):
        if valueTypeCheck(key, Tuple[int, str]):
            key = (str(key[0]), key[1])
        return super().get(key, default)    # type: ignore

    def clear(self):
        for _, node in self.items():
            if hasattr(node, 'ON_DESTROY'):
                try:
                    node.ON_DESTROY()
                except Exception as e:
                    ComfyUILogger.error(f'Error when calling ON_DESTROY for node with id={node.ID}({node.__class__.__qualname__}): {e}, ignored.')
        super().clear()

    def pop(self, key: Union[str, int, Tuple[str, str]], default=None):
        if isinstance(key, tuple):
            if not len(key) in (1, 2):
                raise ValueError(f'Invalid key: {key}, it should be a tuple with 2 elements(node_id, node_cls_name).' )
            id = str(key[0]) if isinstance(key[0], int) else key[0]
            node_type = key[1] if len(key)==2 else None
        else:
            id = str(key) if isinstance(key, int) else key
            node_type = None
        
        node_type = get_cls_name(node_type) if node_type and isinstance(node_type, type) else node_type
        if node_type:
            node = super().pop((id, node_type))
        else:
            node = self.get(id)
            if node:
                node_type = get_cls_name(node)
                super().pop((id, node_type))
        if node:
            if hasattr(node, 'ON_DESTROY'):
                try:
                    node.ON_DESTROY()
                except Exception as e:
                    ComfyUILogger.error(f'Error when calling ON_DESTROY for node with id={node.ID}({node.__class__.__qualname__}): {e}, ignored.')
        return node

class NodeBindingParam(Tuple[str, int], metaclass=NameCheckMetaCls()):
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
    def from_node(self)->"ComfyUINode": # type: ignore
        return NodePool.Instance[self.from_node_id]  # type: ignore
    
    @property
    def output_slot_index(self)->int:
        '''The output slot index of the source node. Wrapper for the second element of the tuple.'''
        return self[1]

class NodeInputs(Dict[str, Union[NodeBindingParam, Any]], metaclass=NameCheckMetaCls()):
    '''
    {param_name: input_value, ...}
    
    There are 2 types of dict value:
        - [node_id(str), output_slot_index(int)]: the input is from another node
        - Any: the input is a value
    '''
    
    def _should_convert_to_bind_type(self, value: list)->bool:
        if isinstance(value, NodeBindingParam):
            return False    # already binding
        if len(value)!=2 or not isinstance(value[0], str) or not isinstance(value[1], int): 
            return False    # not binding
        return True         # seems like a binding, e.g. ['1', 0] means from node_id=1 and output_slot_index=0
    
    def _format_values(self):
        all_param_dict = {}
        all_param_dict.update(self.node_type.INPUT_TYPES().get('required', {}))
        all_param_dict.update(self.node_type.INPUT_TYPES().get('optional', {}))
        all_param_dict.update(self.node_type.INPUT_TYPES().get('hidden', {}))
        
        advanced_node_input_type_def = {}
        if hasattr(self.node_type, '__ADVANCED_NODE_CLASS__') and self.node_type.__ADVANCED_NODE_CLASS__:
            advanced_node_input_type_def = self.node_type.__ADVANCED_NODE_CLASS__._InputFields
            
        for key, value in tuple(self.items()):
            param_type = all_param_dict[key][0] # (type, param info)
            converted = False
            if isinstance(value, list):
                if self._should_convert_to_bind_type(value):
                    self[key] = NodeBindingParam(value)
                    converted = True
            if not converted:
                if isinstance(param_type, type):
                    if hasattr(param_type, "__ComfyLoad__"):
                        self[key] = param_type.__ComfyLoad__(value)
                elif key in advanced_node_input_type_def:
                    param_type = advanced_node_input_type_def[key].origin_type    
                    if isinstance(param_type, type) and hasattr(param_type, "__ComfyLoad__"):
                        self[key] = param_type.__ComfyLoad__(value)
    
    node_id: str
    '''Which node this input belongs to.'''
    node_type_name: str
    '''The origin node type of this input's node.'''
    node_type: Type["ComfyUINode"]
    '''The origin node type of this input's node.'''

    def __init__(self, node_id:str, node_type_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = node_id
        self.node_type_name = node_type_name
        self.node_type = get_node_type_by_name(node_type_name)  # type: ignore
        if not self.node_type:
            raise ValueError(f'Invalid node type name: {node_type_name}.')    
        self._format_values() # make sure all values are Any/FromNodeInput

    def __setitem__(self, __key: str, __value: Union[NodeBindingParam, Any]) -> None:
        if (isinstance(__value, list) and
            len(__value)==2 and 
            not isinstance(__value, NodeBindingParam) and
            isinstance(__value[0], str) and
            isinstance(__value[1], int)):
            
            __value = NodeBindingParam(*__value)
        super().__setitem__(__key, __value)

NodeOutputs_UI: TypeAlias = Dict[str, Dict[str, Any]]
'''All outputs of nodes for ui'''

NodeOutputs: TypeAlias = Dict[str, List[Any]]
'''
All outputs of nodes for execution.
{node id: [output1, output2, ...], ...}
'''

StatusMsgEvent: TypeAlias = Literal['status', 'progress', 'executing', 'executed', 'execution_start', 'execution_error', 'execution_cached', 'execution_interrupted']
'''The status message event type for PromptExecutor. See source/comfyUI/web/scripts/api.js'''

StatusMsgs: TypeAlias = List[Tuple[StatusMsgEvent, Dict[str, Any]]]
'''
The status message type for PromptExecutor.
[(event, {msg_key: msg_value, ...}), ...]
'''

QueueTask: TypeAlias = Tuple[Union[int, float], str, "PROMPT", dict, list]
'''
The type hint for the queue task in execution.PromptQueue's inner items
Items:
    - number (int? float? seems int but i also saw number=float(...), too strange, and idk wtf is this for)
    - prompt_id (str, random id by uuid4)
    - prompt    (for PromptExecutor.execute method)
    - extra_data
    - outputs_to_execute (node ids to be e)
'''

ConvertedConditioning: TypeAlias = List[Dict[str, Any]]
"""
The ConvertedConditioning datatype is a return representation from convert_cond() function in sample.py
It has the following structure:
[
    {
        "some_output_a": Any, 
        "cross_attn": cond_tensor_a,
        "model_conds": {
            "c_crossattn": comfy.conds.CONDCrossAttn(cond_tensor_a)
        }
    },
    {
        "some_output_b": Any, 
        "cross_attn": cond_tensor_b,
        "model_conds": {
            "c_crossattn": comfy.conds.CONDCrossAttn(cond_tensor_b)
        }
    },  
]
Refer to CONDITIONING datatype or convert_cond() docstrings for more information.
"""

class RecursiveExecuteResult(NamedTuple):
    '''result return type for _recursive_execute in execution.py'''
    
    success: bool
    '''whether the execution is successful or not'''
    error_details: Optional[dict]
    '''the error details if the execution is failed'''
    exception: Optional[Exception]
    '''the exception if the execution is failed'''

@dataclass
class ValidateInputsResult:
    '''dataclass for the result of `validate_inputs` function.'''
    
    success: bool
    '''whether the validation is successful or not'''
    errors: List[dict]
    '''list of error messages'''
    node_id: str
    '''the node id that is being validated'''

    def __getitem__(self, item: int):
        if item not in (0, 1, 2, 3):
            raise IndexError(f"Index out of range: {item}")
        if item == 0:
            return self.success
        if item == 1:
            return self.errors
        if item == 2:
            return self.node_id
        if item == 3:
            return self.adapter

@dataclass
class ValidatePromptResult:
    '''dataclass for the result of `validate_prompt` function.'''
    
    result: bool
    '''whether the validation is successful or not'''
    errors: Optional[dict]
    '''the error messages if the validation failed'''
    nodes_with_good_outputs: List[str]
    '''list of output node ids that passed the validation.'''
    node_errors: Dict[str, dict]
    '''dict of node_id: error messages'''

    _prompt: dict
    '''The real input prompt, a dictionary (converted from json)'''
    _prompt_id: str
    '''unique id of the prompt, a random string by uuid4. This can be None to make it compatible with old comfyUI codes.'''
    _formatted_prompt: Optional["PROMPT"] = None
    '''The properly formatted prompt, in `PROMPT` type'''
    
    @property
    def formatted_prompt(self)->"PROMPT":
        from .hidden import PROMPT
        if not self._formatted_prompt:
            if not isinstance(self._prompt, PROMPT):
                self._formatted_prompt = PROMPT(self._prompt, id=self._prompt_id)
            else:
                self._formatted_prompt = self._prompt
        return self._formatted_prompt
    
    def __getitem__(self, item: int):
        if item not in (0, 1, 2, 3, 4):
            raise IndexError(f"Index out of range: {item}. It should be in [0, 1, 2, 3, 4]")
        if item == 0:
            return self.result
        if item == 1:
            return self.errors
        if item == 2:
            return self.nodes_with_good_outputs
        if item == 3:
            return self.node_errors
        if item == 4:
            return self.formatted_prompt

_T = TypeVar('_T')
_MT = TypeVar('_MT') 
class NODE_MAPPING(Dict[Tuple[str, str], _MT], Generic[_MT]):
    '''
    Node mappings dict, e.g. `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`,... in `nodes.py`.
    This class allows you to find node in this mapping with just `node_type_name` or (node_type_name, namespace).
    '''
    
    class NodeMappingKey(str):
        '''
        This class acts as the key when u iterating `NODE_MAPPING` dict.
        It is for the compatibility with old code.
        '''
        
        cls_name: str
        namespace: str
        origin_tuple: Tuple[str, str]
        
        def __new__(cls, cls_name_and_namespace: Tuple[str, str]):
            ins = super().__new__(cls, cls_name_and_namespace[0])
            ins.cls_name = cls_name_and_namespace[0]
            ins.namespace = cls_name_and_namespace[1]
            ins.origin_tuple = cls_name_and_namespace
            return ins
        
        def __len__(self):
            return 2
        
        def __iter__(self):
            return iter(self.origin_tuple)

        def __getitem__(self, key) -> str:
            return self.origin_tuple[key]
        
        def __hash__(self):
            return hash(self.cls_name)
        
        def __eq__(self, other: Union[Tuple[str, str], str]):
            if isinstance(other, tuple):
                if len(other)!=2:
                    return False
                return self.cls_name == other[0] and self.namespace == other[1]
            elif isinstance(other, str):
                return self.cls_name == other
            return False

        def __str__(self):
            return self.cls_name
        
        def __repr__(self) -> str:
            return self.cls_name
    
    def __class_getitem__(cls, t: _T) -> 'Type[NODE_MAPPING[_T]]':
        # this is just acting as a type hint
        return cls  # type: ignore
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in tuple(super().keys()):
            if isinstance(key, str):
                super().__setitem__((key, ""), super().pop(key))    # `""` means default namespace
    
    def __getitem__(self, key: Union[str, type, Tuple[Union[str, type], Union[str, None]]]) -> _MT:
        if isinstance(key, (list, tuple)):
            if len(key) not in (1,2):
                raise ValueError(f'Invalid key: {key}, it should be a tuple with 2 elements(node_type_name, namespace).')
            node_type_name = key[0]
            node_namespace = key[1] if len(key)==2 else None
        else:
            node_type_name = key
            node_namespace = None
        if isinstance(node_type_name, type):
            node_type_name = get_cls_name(node_type_name)
        node_namespace = node_namespace.strip() if node_namespace else None
        node_namespace = "" if node_namespace is None else node_namespace   # `""` means default namespace
        try:
            return super().__getitem__((node_type_name, node_namespace))
        except KeyError as e:
            if node_namespace:   # namepsace is specified but not found
                raise e
            for key in tuple(super().keys()):
                if key[0] == node_type_name:
                    return super().__getitem__(key)
            raise KeyError(f'Node not found: {node_type_name} in namespace {node_namespace}.')
    
    def __setitem__(self, key: Union[str, type, Tuple[Union[str, type], Union[str, None]]], value: _MT) -> None:
        if isinstance(key, (list, tuple)):
            if len(key) not in (1,2):
                raise ValueError(f'Invalid key: {key}, it should be a tuple with 2 elements(node_type_name, namespace).')
            node_type_name = key[0]
            node_namespace = key[1] if len(key)==2 else None
        else:
            node_type_name = key
            node_namespace = None
        if isinstance(node_type_name, type):
            node_type_name = get_cls_name(node_type_name)
        node_namespace = node_namespace.strip() if node_namespace else None
        node_namespace = "" if node_namespace is None else node_namespace   # `""` means default namespace
        super().__setitem__((node_type_name, node_namespace), value)

    def keys(self):
        return self.__iter__()

    def __iter__(self):
        for key in super().keys():
            yield NODE_MAPPING.NodeMappingKey(key)

@attrs
class SamplingCallbackContext:
    '''context during sampling.'''
    
    noise: torch.Tensor = attrib()
    '''The noise tensor for sampling.'''
    step_index: int = attrib()
    '''The current step index'''
    denoised: torch.Tensor = attrib()
    '''the denoised latent'''
    total_steps: int = attrib()
    '''The total steps of sampling.'''
    
    # region deprecated
    @property
    @deprecated(reason='use `noise` instead.')
    def x(self)->torch.Tensor:
        '''The noise tensor for sampling.'''
        return self.noise
    
    @property
    @deprecated(reason='use `step_index` instead')
    def i(self)->int:
        return self.step_index
    # endregion
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)

SamplerCallback: TypeAlias = Union[
    Callable[[], Any],  # no arguments
    Callable[[SamplingCallbackContext], Any],    # pass context
    Callable[[int, torch.Tensor, torch.Tensor, int], Any]   # pass (i, denoised, x, total_steps)
]
'''callback when a inference step is finished''' 


__all__ = ['InferenceContext', 'SamplingCallbackContext', 'SamplerCallback',
                
            'NodePool', 'NodeBindingParam', 'NodeInputs', 'NodeOutputs', 'NodeOutputs_UI',
            
            'StatusMsgEvent', 'StatusMsgs', 'QueueTask', 'ConvertedConditioning', 
            
            'RecursiveExecuteResult', 'ValidateInputsResult', 'ValidatePromptResult',
            
            'NODE_MAPPING']
