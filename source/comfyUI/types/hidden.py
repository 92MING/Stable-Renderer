from typing import (Union, Annotated, Literal, Optional, Any, Type, Dict, TYPE_CHECKING, ClassVar,
                    TypeVar, Tuple, List)
from attr import attrs, attrib
from abc import ABC, abstractmethod, ABCMeta
from deprecated import deprecated
from common_utils.global_utils import GetOrCreateGlobalValue
from common_utils.type_utils import NameCheckMetaCls, get_cls_name
from common_utils.decorators import prevent_re_init

from ._utils import *
from .basic import AnnotatedParam

if TYPE_CHECKING:
    from engine.static.texture import Texture
    from .runtime import InferenceContext, NodeInputs
    from .node_base import ComfyUINode


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


_AllPrompts: dict = GetOrCreateGlobalValue('_all_prompts', dict)
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
    
    def __new__(cls, *args, id:str=None, **kwargs): # type: ignore
        if id is not None and id in _AllPrompts:
            val = _AllPrompts[id]
        else:
            val = super().__new__(cls)
        return val
    
    def __init__(self, *args, id:str=None, **kwargs): # type: ignore
        super().__init__(*args, **kwargs)
        self._id = id
        self._format_prompt() # tidy up types
        self._setitem_formatting = True
    
    def __setitem__(self, key: str, val: Dict[Literal['inputs', 'class_type', 'is_changed'], Any]):
        if not self._setitem_formatting:
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
    _setitem_formatting: bool =False
    '''enable after init, means start checking the format of inputs whtn calling prompt's __setitem__.'''
    
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
        if not isinstance(node, str):
            node = node.ID
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
    
    def destroy(self):
        self.clear()
        _AllPrompts.pop(self.id, None) if self.id is not None else None


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

@attrs
class FrameData(HIDDEN):
    '''The prompt for submitting to ComfyUI during engine's runtime.'''

    colorMap: "Texture" = attrib(default=None)
    '''color of the current frame'''
    idMap: "Texture" = attrib(default=None)
    '''
    ID of each pixels in the current frame. If the pixel is not containing any object, the id is 0,0,0,0.
    ID data has two possible combinations:
        - (objID, materialID, texX, texY): in rendering mode
        - (baking pixel index, materialID, texX, texY): in baking mode
    '''
    posMap: "Texture" = attrib(default=None)
    '''position of the origin 3d vertex in each pixels in the current frame'''
    normalAndDepthMap: "Texture" = attrib(default=None)
    '''
    normal and depth of the origin 3d vertex in each pixels in the current frame.
    Format: (nx, ny, nz, depth)
    '''
    noiseMap: "Texture" = attrib(default=None)
    '''noise of each vertex in each pixels in the current frame'''
    
    @classmethod
    def GetHiddenValue(cls, context):
        return context.frame_data


@attrs
class BakingData(HIDDEN):
    '''TODO: Runtime data during baking.'''
    
    @classmethod
    def GetHiddenValue(cls, context):
        return context.baking_data
    