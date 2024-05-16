import torch
from torch import Tensor
from typing import (Union, Annotated, Literal, Optional, Any, Type, Dict, TYPE_CHECKING, ClassVar,
                    TypeVar, Tuple, List)
from attr import attrs, attrib
from abc import ABC, abstractmethod, ABCMeta
from deprecated import deprecated
from einops import rearrange

from common_utils.type_utils import NameCheckMetaCls, get_cls_name
from common_utils.decorators import prevent_re_init, class_or_ins_property
from common_utils.global_utils import is_dev_mode, is_verbose_mode
from common_utils.debug_utils import ComfyUILogger
from ._utils import *
from .basic import AnnotatedParam, IMAGE, MASK, LATENT

if TYPE_CHECKING:
    from engine.static.texture import Texture
    from engine.static.corrmap import IDMap, CorrespondMap
    from comfyUI.execution import PromptExecutor
    from .runtime import NodeInputs
    from .node_base import ComfyUINode
    from .runtime import InferenceOutput, StatusMsgs
    from common_utils.stable_render_utils import SpriteInfos, EnvPrompt

_hidden_meta = NameCheckMetaCls(ABCMeta)
_HT = TypeVar('_HT', bound='HIDDEN')

class HIDDEN(ABC, metaclass=_hidden_meta):
    '''
    The base class for all special hidden types. Class inherited from this class will must be treated as hidden type in node system.
    
    You could also define hidden types by naming params with prefix `_`.
    But all hidden types have special meanings/ special behaviors in ComfyUI,
    so it is recommended to use this class to define hidden types.
    '''
    def __class_getitem__(cls, tp: type):
        return Annotated[tp, AnnotatedParam(origin=tp, tags=['hidden'])]
    
    @classmethod
    def _SubClses(cls):
        sub_clses = [cls, ]
        for subcls in cls.__subclasses__():
            sub_clses.extend(subcls._SubClses())
        return sub_clses
    
    @classmethod
    def _ClsName(cls):
        return get_comfy_name(cls)
    
    @staticmethod
    def FindHiddenClsByName(cls_name: str):
        all_subclses = HIDDEN._SubClses()[1:]   # exclude the base class itself
        for subcls in all_subclses:
            if subcls._ClsName() == cls_name:
                return subcls
        return None
    
    @classmethod
    @abstractmethod
    def GetHiddenValue(cls: Type[_HT], context: "InferenceContext")->Optional[_HT]:   # type: ignore
        '''All hidden types should implement this method to get the real value from current inference context.'''
        raise NotImplementedError

@prevent_re_init
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
        from .runtime import NodeInputs
        for node_id, node_info_dict in self.items():
            if 'inputs' in node_info_dict:
                if not isinstance(node_info_dict['inputs'], NodeInputs):
                    node_cls_name = node_info_dict.get('class_type')
                    if not node_cls_name:
                        raise ValueError(f'Node class {node_id} has no class type.')
                    else:
                        node_cls_name = get_cls_name(node_cls_name) if isinstance(node_cls_name, type) else node_cls_name
                    node_info_dict['inputs'] = NodeInputs(node_id, node_cls_name, node_info_dict['inputs'])
            if 'class_type' in node_info_dict:
                if isinstance(node_info_dict['class_type'], type):
                    node_info_dict['class_type'] = get_cls_name(node_info_dict['class_type'])
    
    def __init__(self, *args, id:str=None, **kwargs): # type: ignore
        super().__init__(*args, **kwargs)
        self._id = id
        self._format_prompt() # tidy up types
        self._setitem_check_formatting = True
    
    def __setitem__(self, key: str, val: Dict[Literal['inputs', 'class_type', 'is_changed'], Any]):
        if not self._setitem_check_formatting:
            return super().__setitem__(key, val)
        else:
            if not isinstance(val, dict):
                return super().__setitem__(key, val)
            else:
                from .runtime import NodeInputs
                for dict_key, dict_val in val.items():
                    if dict_key == 'inputs' and not isinstance(dict_val, NodeInputs):
                        cls_type = val.get('class_type')
                        node_cls_name = get_cls_name(cls_type) if isinstance(cls_type, type) else cls_type
                        if not node_cls_name:
                            raise ValueError(f'Node class {key} has no class type.')
                        val['inputs'] = NodeInputs(key, node_cls_name, dict_val)
                    elif dict_key == 'class_type':
                        if isinstance(dict_val, type):
                            val['class_type'] = get_cls_name(dict_val)
                    elif dict_key == 'is_changed':
                        pass    # maybe do some check here, so far, do nothing
    
    _id: str
    '''a unique random id generated by uuid4. This can be None, so as to compatible with old code.'''
    _setitem_check_formatting: bool =False
    '''enable after init, means start checking the format of inputs when calling prompt's __setitem__.'''
    
    @property
    def id(self)->str:
        return self._id
    
    @property
    def links(self)->Dict[str, Dict[int, List[Tuple[str, str]]]]:
        '''
        Returns all nodes' connections.
        {from_node_id: {output_slot: [(to_node_id, to_param),]}}
        '''
        links = {}
        for node_id in self:
            if node_input_dict := self.get_node_inputs(node_id):
                for input_param_name, bindings in node_input_dict.links.items():
                    for binding in bindings:
                        from_node_id = binding.from_node_id
                        from_output_slot = binding.output_slot_index
                        if from_node_id not in links:
                            links[from_node_id] = {}
                        if from_output_slot not in links[from_node_id]:
                            links[from_node_id][from_output_slot] = []
                        links[from_node_id][from_output_slot].append((node_id, input_param_name))
        return links
                    
    @classmethod
    def GetHiddenValue(cls, context: "InferenceContext"):
        '''Get the real value from current inference context.'''
        return context.prompt
    
    def get_node_inputs(self, node: Union[str, "ComfyUINode"])->Optional["NodeInputs"]:
        '''
        The input data for the node. The wrapper for `inputs` key.
        Each node in prompt dict must have a input data, so this property must be available.
        '''
        from .node_base import ComfyUINode
        
        if isinstance(node, ComfyUINode):
            node = node.ID
        elif not isinstance(node, (str, int)):
            raise TypeError(f'Node must be str or ComfyUINode, but got {type(node)}.')
        else:
            node = str(node)
        
        if node not in self:
            raise KeyError(f'Node {node} not found in prompt dict.')
        
        return self[node].get('inputs')
    
    def get_node_type_name(self, node: Union[str, "ComfyUINode"])->str:
        '''
        The class name of the node. The wrapper for `class_type` key.
        Each node in prompt dict must have a class name, so this property must be available.
        '''
        if not isinstance(node, str):
            node = node.ID
        return self[node]['class_type']   # type: ignore
    
    def get_node_type(self, node: Union[str, "ComfyUINode"])->Type["ComfyUINode"]:
        '''return the class type of the node. The wrapper for `class_type` key.'''
        cls_type = self.get_node_type_name(node)
        from comfyUI.nodes import NODE_CLASS_MAPPINGS
        return NODE_CLASS_MAPPINGS[cls_type]
    
    def get_is_changed_validation_val(self, node: Union[str, "ComfyUINode"])->Optional[Any]:
        '''
        The value for identifying whether the input is changed(if u have defined `IS_CHANGED` method).
        Wrapper for `is_changed` key.
        
        This method returns optional value, because the `IS_CHANGED` method is not necessary to be defined in comfyUI nodes.
        '''
        if not isinstance(node, str):
            node = node.ID
        return self[node].get('is_changed', None)

class EXTRA_DATA(Dict[str, Any], HIDDEN):
    '''the dictionary `extra_data` that u pass to PromptExecutor.execution'''
    
    @classmethod
    def GetHiddenValue(cls: type[_HT], context: "InferenceContext"):
        return context.extra_data
    
class EXTRA_PNG_INFO(Dict[str, Any], HIDDEN):
    '''Extra information for saving png file.'''
    
    __ComfyName__: ClassVar[str] = 'EXTRA_PNGINFO'
    
    @classmethod
    def GetHiddenValue(cls: Type[_HT], context: "InferenceContext")->Optional[_HT]:   # type: ignore
        '''All hidden types should implement this method to get the real value from current inference context.'''
        if 'extra_pnginfo' in context.extra_data:
            return context.extra_data['extra_pnginfo']
        return None

@deprecated(reason='unique id of node can be gotten by node.ID directly. It is not necessary to use this type anymore.')
class UNIQUE_ID(str, HIDDEN):
    '''The unique ID of current running node.'''
    
    @classmethod
    def GetHiddenValue(cls: Type[_HT], context: "InferenceContext")->Optional[_HT]:
        cur_node = context.current_node_id
        if cur_node is not None:
            return cur_node.ID  # type: ignore
        return None

class EnvPrompts(list["EnvPrompt"], HIDDEN):
    '''
    List of environment prompts for the current frame(s).
    This type is for being an annotation for node system.
    '''
    @classmethod
    def GetHiddenValue(cls, context: "InferenceContext"):
        if not context or not context.engine_data:
            return []
        return context.engine_data.env_prompts

class CorrespondMaps(dict[tuple[int, int], "CorrespondMap"], HIDDEN):
    '''The correspondence maps for the current frame(s).'''
    
    @classmethod
    def GetHiddenValue(cls, context: "InferenceContext"):
        if not context or not context.engine_data:
            return {}
        return context.engine_data.correspond_maps

@attrs
class EngineData(HIDDEN):
    '''
    Most useful data from rendering engine will be passing to ComfyUI through this data class.
    Note that an engine data may include multiple frames.
    '''
    
    frame_indices: List[int] = attrib(default=0, kw_only=True)
    '''The indices of the frames in this engine data obj.'''
    @property
    def frame_count(self)->int:
        '''The number of frames in this engine data obj.'''
        return len(self.frame_indices)
    
    sprite_infos: "SpriteInfos" = attrib(factory=dict, kw_only=True)
    '''all sprite infos. This is actually a dictionary with {spriteID: info} pairs'''
    
    color_maps: Optional[IMAGE] = attrib(default=None, kw_only=True)
    '''
    Color of the current frame(s).
    shape=(N, H, W, 3). dtype=float32, range=[0, 1]
    When engine data is created, the data will be formatted into the correct shape and dtype.
    Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    id_maps: Optional["IDMap"] = attrib(default=None, kw_only=True)
    '''
    ID of each pixels in the current frame(s). If the pixel is not containing any object, the id is 0,0,0,0.
    Format: (spriteID, materialID, baking map index, vertexID): in baking mode
    shape = (N, H, W, 4), dtype = int32
    Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    pos_maps: IMAGE = attrib(default=None, kw_only=True)
    '''
    Position of the origin 3d vertex in each pixels in the current frame(s).
    This can be used for some special algos to trace the movement of a vertex.
    Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    noise_maps: Optional[LATENT] = attrib(default=None, kw_only=True)
    '''
    Noise of each vertex in each pixels in the current frame
    The latent for inputting to stable diffusion. Note that each 8*8 is merged into 1*1
    Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    normal_maps: Optional[IMAGE] = attrib(default=None, kw_only=True)
    '''
    Normal map of the current frame(s). shape=(N, H, W, 3)
    This can be used as the input of controlnet.
    Note: Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    depth_maps: Optional[IMAGE] = attrib(default=None, kw_only=True)
    '''
    Depth map of the current frame(s). shape=(N, H, W).
    This can be used as the input of controlnet.
    Note: Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    canny_maps: Optional[IMAGE] = attrib(default=None, kw_only=True)
    '''
    Canny edge map of the current frame(s). shape=(N, H, W).
    This can be used as the input of controlnet.
    Note: Note: You could also directly pass `Texture` or `tensor` to it.
    '''
    
    masks: MASK = attrib(default=None, kw_only=True)
    '''
    The mask of the current frame(s). shape=(H, W).
    Mask is the inverse of alpha channel in color map, i.e. 1-alpha.
    '''
    
    env_prompts: EnvPrompts = attrib(factory=EnvPrompts, kw_only=True)
    '''
    This field is actually list[EnvPrompt], which is a list of environment prompts for the current frame(s),
    i.e. when frame count =1, it is a list of one EnvPrompt.
    Note: you can init this field with str, single EnvPrompt, or list of str/EnvPrompt.
    '''
    
    correspond_maps: CorrespondMaps = attrib(factory=dict, kw_only=True)
    '''All correspondence maps for the current frame(s).'''

    def __attrs_post_init__(self):
        from common_utils.stable_render_utils import EnvPrompt
        
        if isinstance(self.env_prompts, str):
            self.env_prompts = [EnvPrompt(prompt=self.env_prompts),]   # type: ignore
        elif isinstance(self.env_prompts, EnvPrompt):
            self.env_prompts = [self.env_prompts, ] # type: ignore
        elif isinstance(self.env_prompts, (list, tuple)):
            prompts = []
            for prompt in self.env_prompts:
                if isinstance(prompt, str):
                    prompts.append(EnvPrompt(prompt=prompt))
                elif isinstance(prompt, EnvPrompt):
                    prompts.append(prompt)
                else:
                    raise ValueError(f'Invalid type of env_prompts: {type(prompt)}')
            self.env_prompts = prompts  # type: ignore
        
    @classmethod
    def GetHiddenValue(cls, context):
        return context.engine_data


@attrs
class InferenceContext(HIDDEN): # InferenceContext is also a hidden value
    '''
    Inference context is the context across the whole single inference process.
    For each `PromptExecutor.execute` call, a new InferenceContext will be created,
    and each function during the execution will receive this context as the first argument.
    
    This is also the context for hidden types to find out their hidden values during execution,
    by passing context to the `GetHiddenValue` function.
    
    In engine's rendering loop, this is also the final output in DiffusionManager.SubmitPrompt.
    Useful values can be gotten directly from this context.
    '''
    
    @class_or_ins_property  # type: ignore
    def _executor(cls_or_ins)-> 'PromptExecutor':
        '''The prompt executor instance.'''
        from comfyUI.execution import PromptExecutor
        return PromptExecutor() # since the PromptExecutor is a singleton, it is safe to create a new instance here
    
    prompt:"PROMPT" = attrib()
    '''The prompt of current execution. It is a dict containing all nodes' input for execution.'''
    
    extra_data: dict[str, Any] = attrib(factory=dict)
    '''Extra data for the current execution.'''
    
    _current_node_id: Optional[str] = attrib(default=None, alias='current_node_id')
    '''The current node for this execution.'''
    
    outputs: Dict[str, List[Any]] = attrib(factory=dict)
    '''
    The outputs of the nodes for execution. {node id: [output1, output2, ...], ...}
    Note that in PromptExecutor, the `outputs` is a `ComfyCacheDict`, while here it is a normal dict.
    '''
    
    outputs_ui: Dict[str, Dict[str, Any]] = attrib(factory=dict)
    '''
    The outputs of the nodes for ui. {node id: {output_name: [output_value1, ...], ...}, ...}
    Note that in PromptExecutor, the `outputs_ui` is a `ComfyCacheDict`, while here it is a normal dict.
    '''
    
    to_be_executed: List[Tuple[int, str]] = attrib(factory=list)
    '''node ids waiting to be executed. [(order, node_id), ...]'''
    
    executed_node_ids: set[str] = attrib(factory=set)
    '''The executed nodes' ids.'''
    
    engine_data: Optional["EngineData"] = attrib(default=None)
    '''The data passed from engine to comfyUI for this execution.'''
    
    status_messages: 'StatusMsgs' = attrib(factory=list)
    '''The status messages for current execution. [(event, {msg_key: msg_value, ...}), ...]'''
    
    success:bool = attrib(default=False)
    '''Whether the execution is successful.'''
    
    final_output: "InferenceOutput" = attrib(default=None)
    '''the final output for `DiffusionManager.SubmitPrompt`.'''
    
    def node_is_waiting_to_execute(self, node_id: Union[str, int])->bool:
        node_id = str(node_id)
        for (_, id) in self.to_be_executed:
            if id == node_id:
                return True
        return False
    
    def remove_from_execute_waitlist(self, node_id: Union[str, int]):
        node_id = str(node_id)
        for (_, id) in tuple(self.to_be_executed):
            if id == node_id:
                self.to_be_executed.remove((_, id))
            
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
            node_id = value.ID
            node_type_name = self.prompt[node_id]['class_type']
            if node_type_name != value.NAME:
                ComfyUILogger.warning(f'The given node is not the same as the node in the prompt: {node_type_name} != {value.NAME}')
            value = node_id
            
        if isinstance(value, str):
            if value != self._current_node_id:
                self._current_node_id = value
                if is_dev_mode() and is_verbose_mode():
                    if self.current_node_cls_name:
                        ComfyUILogger.debug(f'Current running node id is set to: {value}({self.current_node_cls_name})')
                    else:
                        ComfyUILogger.debug(f'Current running node id is set to: {value}')
        else:
            raise ValueError(f'Invalid current node id type: {value}')
        
    @property
    def current_node_cls_name(self)->Optional[str]:
        '''return the class name of current node.'''
        if not self.current_node_id:
            return None
        try:
            return self.prompt[self.current_node_id]['class_type']
        except KeyError as e:
            if is_dev_mode():
                raise e
            else:
                return None
        
    @property
    def current_node_cls(self)->Optional[Type["ComfyUINode"]]:
        '''return the origin type of current node.'''
        if not self.current_node_cls_name:
            return None
        return get_node_cls_by_name(self.current_node_cls_name)
    
    @property
    def current_node(self)->Optional["ComfyUINode"]:
        '''return the current node instance.'''
        from .runtime import NodePool
        if not self.current_node_id:
            return None
        return NodePool.Instance().get_or_create(self.current_node_id, self.current_node_cls_name)  # type: ignore

    @current_node.setter
    def current_node(self, value: Union[str, int, "ComfyUINode"]):
        self.current_node_id = value

    @classmethod
    def GetHiddenValue(cls, context: "InferenceContext"):
        return context  # just return the context itself


__all__ = ['HIDDEN', 'PROMPT', 'EXTRA_DATA', 'EXTRA_PNG_INFO', 'UNIQUE_ID', 'EngineData', 'InferenceContext', 'EnvPrompts', 'CorrespondMaps']