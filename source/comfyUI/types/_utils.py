from typing import (Union, TypeAlias, Annotated, Literal, Optional, get_origin, get_args, ForwardRef, 
                    Any, Set, TYPE_CHECKING, Type, Dict, Tuple)
from inspect import Parameter
from enum import Enum
from pathlib import Path

from common_utils.type_utils import get_cls_name, subClassCheck
from common_utils.global_utils import GetGlobalValue, GetOrCreateGlobalValue

if TYPE_CHECKING:
    from .basic import NodeInputParamType
    from .node_base import ComfyUINode


_SPECIAL_TYPE_NAMES = {
    str: "STRING",
    bool: "BOOLEAN",
    int: "INT",
    float: "FLOAT",
    Path: "PATH",
}

def get_comfy_name(tp: Any, inner_type: Optional[Union[type, str]]=None)->str:
    '''Get the type name for comfyUI. It can be any value or type, or name'''
    type_name = get_comfy_type_definition(tp, inner_type)
    if isinstance(type_name, list):
        type_name = 'COMBO'
    return type_name

def get_comfy_type_definition(tp: Union[type, str, TypeAlias], inner_type: Optional[Union[type, str]]=None)->Union[str, list]:
    '''
    return the type name or list of value for COMBO type.
    e.g. int->"INT", Literal[1, 2, 3]->[1, 2, 3]
    '''
    if inner_type is not None:
        tp_definition = get_comfy_type_definition(inner_type)
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
            return [arg for arg in tp.__args__] # type: ignore
        elif origin == Annotated:
            args = get_args(tp)
            for arg in args[1:]:
                from .basic import AnnotatedParam
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

def get_input_param_type(param: Parameter, 
                         tags: Optional[Set[str]]=None)->"NodeInputParamType":
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
    
    # tags contains 'hidden' means it is hidden
    if tags:   
        if 'hidden' in tags:
            return 'hidden'
    
     # param name starts with '_'
    input_param_name = param.name
    if input_param_name.startswith('_'):
        return 'hidden'
    
    anno = param.annotation
    anno_origin = get_origin(anno)
    
    from .hidden import HIDDEN
    from .basic import AnnotatedParam
    
    # type annotation is 'hidden' or is a subclass of HIDDEN
    if anno_origin:
        if anno_origin == Annotated:
            args = get_args(anno)
            if isinstance(args[0], str):
                if args[0].lower() == 'hidden': 
                    return 'hidden'
            if isinstance(args[0], type):
                if issubclass(args[0], HIDDEN): 
                    return 'hidden'
            if isinstance(args[0], AnnotatedParam):
                if subClassCheck(args[0].origin_type, HIDDEN):
                    return 'hidden'
            if len(args)>=2 and isinstance(args[1], AnnotatedParam):
                if subClassCheck(args[1].origin_type, HIDDEN):
                    return 'hidden'
    if isinstance(anno, AnnotatedParam):
        if subClassCheck(anno.origin_type, HIDDEN):
            return 'hidden'
    if isinstance(anno, type):
        if subClassCheck(anno, HIDDEN):
            return 'hidden'
    if isinstance(anno, str):
        if anno.lower() == 'hidden':
            return 'hidden'
    
    # if has default value, it is optional
    if param.default != Parameter.empty:    
        return 'optional'
    
    return 'required'  

def get_node_cls_by_name(name: str, namespace: Optional[str]=None, init_nodes_if_not_yet=False)->Optional[Type["ComfyUINode"]]:
    '''this method will only return corret result when custom nodes are initialized'''
    from comfyUI.nodes import NODE_CLASS_MAPPINGS
    try:
        return NODE_CLASS_MAPPINGS[(name, namespace)]
    except KeyError:
        if not GetGlobalValue("__COMFYUI_CUSTOM_NODES_INITED__", False):
            if init_nodes_if_not_yet:
                from comfyUI.nodes import init_custom_nodes
                init_custom_nodes()
                try:
                    return NODE_CLASS_MAPPINGS[(name, namespace)]
                except KeyError:
                    return None 
        return None


__all__ = ['get_comfy_name', 'get_comfy_type_definition', 'get_input_param_type', 'get_node_cls_by_name']


_comfy_get_input_type_name_cache: Dict[Tuple[int, str], str] = GetOrCreateGlobalValue("__COMFY_GET_INPUT_TYPE_NAME_CACHE__", dict)

def get_comfy_node_input_type(node_type: Union["ComfyUINode", Type["ComfyUINode"]], input_name: str)->str:
    '''return the type of the input parameter of the node.'''
    if not isinstance(node_type, type):
        node_type = node_type.__class__
        
    node_type_id = id(node_type)
    if (node_type_id, input_name) in _comfy_get_input_type_name_cache:
        return _comfy_get_input_type_name_cache[(node_type_id, input_name)]
    
    input_param_dict = node_type.INPUT_TYPES()
    all_input_params = {}
    all_input_params.update(input_param_dict.get("required", {}))
    all_input_params.update(input_param_dict.get("optional", {}))
    all_input_params.update(input_param_dict.get("hidden", {}))
    if input_name not in all_input_params:
        raise ValueError(f"Input name {input_name} not found in node {node_type}.")
    t_name = all_input_params[input_name][0]
    if isinstance(t_name, (list, tuple)):
        t_name = "COMBO"
    _comfy_get_input_type_name_cache[(node_type_id, input_name)] = t_name
    return t_name

def check_input_param_is_list_type(param_name:str, node_class: Type["ComfyUINode"])->bool:
    '''check wether the node is labelled with `INPUT_IS_LIST` or the input parameter is tagged with `list`.'''
    if hasattr(node_class, "INPUT_IS_LIST") and node_class.INPUT_IS_LIST:
        return True
    if hasattr(node_class, '__ADVANCED_NODE_CLASS__') and node_class.__ADVANCED_NODE_CLASS__:
        input_infos = node_class.__ADVANCED_NODE_CLASS__._InputFields
        return ('list' in input_infos[param_name].tags) if param_name in input_infos else False
    return False

def check_input_param_is_lazy(param_name: str, node_class: Union["ComfyUINode", Type["ComfyUINode"]]):
    '''check whether the input parameter is lazy type.'''
    if not hasattr(node_class, "LAZY_INPUTS") or not node_class.LAZY_INPUTS:
        return False
    lazy_inputs = node_class.LAZY_INPUTS
    return param_name in lazy_inputs



__all__.extend(['get_comfy_node_input_type', 'check_input_param_is_list_type', 'check_input_param_is_lazy'])