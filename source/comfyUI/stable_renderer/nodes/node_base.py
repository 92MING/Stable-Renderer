'''
Base class for all nodes.
All sub classes will be registered automatically.
'''

import re
from inspect import isabstract, signature, Parameter, Signature
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Sequence, Literal, Any, Tuple
from enum import Enum
from collections import OrderedDict

from comfyUI.stable_renderer.utils.decorators import classproperty
from comfyUI.stable_renderer.utils.comfy_base_types import *
from comfyUI.stable_renderer.utils.comfy_base_types import UIType

def get_input_param_type(key: str, param: Parameter)->Literal['required', 'optional', 'hidden']:
    '''
    Get the input parameter type for registration.
    
    Conditions:
        * hidden: param name starts with '_', e.g. def f(_x:int)
        * optional: param default is None, e.g. def f(x:int = None)
        * required: others
    '''
    if key.startswith('_'):
        return 'hidden'
    if param.default is None:
        return 'optional'   # only when default=None will be considered as optional
    return 'required'  

def _get_proper_param_name(param_name:str):
    while param_name.startswith('_'):
        param_name = param_name[1:]
    return param_name

def _pack_param(sig: Signature, args, kwargs)-> Optional[OrderedDict[str, Any]]:
    '''Pack the args and kwargs into a dict, according to the signature of the function. Return None if not suitable.'''
    var_args_field_name, var_kwargs_field_name = None, None
    for name, param in sig.parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            var_args_field_name = name
        elif param.kind == Parameter.VAR_KEYWORD:
            var_kwargs_field_name = name
        if var_args_field_name and var_kwargs_field_name:
            break
        
    parameters = sig.parameters
    packed_params = OrderedDict()
    if var_args_field_name:
        packed_params[var_args_field_name] = []
    if var_kwargs_field_name:
        packed_params[var_kwargs_field_name] = {}
    
    for k, v in kwargs.items():
        if k in parameters:
            packed_params[k] = v
        elif not var_kwargs_field_name:
            return None # means the kwargs contains some unexpected params, not suitable for this function
        else:
            packed_params[var_kwargs_field_name][k] = v
    
    args_copy = list(args)
    for name, param in parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:  # *args param
            for arg in args_copy:
                packed_params[name].append(arg) # already created a list in the first step, i.e. name == self.var_args_field_name
                args_copy = args_copy[1:]
        elif param.kind == Parameter.VAR_KEYWORD:   # **kwargs param
            for (k, v) in tuple(packed_params.items()):
                packed_params[name][k] = v  # already created a dict in the first step, i.e. name == self.var_kwargs_field_name
        else:   # normal param
            if len(args_copy)>0:
                packed_params[name] = args_copy[0]
                args_copy = args_copy[1:]
            else:
                if param.default == Parameter.empty:
                    return None # means the args are not enough for this function, not suitable for this function
                packed_params[name] = param.default
    
    if var_args_field_name:
        packed_params[var_args_field_name] = tuple(packed_params[var_args_field_name])
        
    return packed_params

def _return_proper_value(values: Tuple[Any, ...]):
    '''Proper the return types to be suitable for ComfyUI.'''
    vals = []
    for val in values:
        if isinstance(val, Enum):
            vals.append(val.name)
        elif isinstance(val, LATENT):
            vals.append(val.origin_dict)
        else:
            vals.append(val)
    return tuple(vals)    

class NodeBase(ABC):
    
    IsAbstract: ClassVar[Optional[bool]] = None
    '''
    Define whether this node is abstract or not. If None, it will be determined automatically.
    Abstract nodes will not be registered.
    '''
    
    Category: ClassVar[Optional[str]] = None
    '''The category of this node. This is for grouping nodes in the UI.'''
    
    Name: ClassVar[Optional[str]] = None
    '''The user friendly name of this node. This is for registration. If not defined, the class name(split by camel case & underscore) will be used.'''
    
    # region internal use
    @staticmethod  # type: ignore
    def _AllSubclasses(non_abstract_only: bool = True):
        all_clses = list(NodeBase.__subclasses__())
        subclses = set()
        while all_clses:
            cls = all_clses.pop()
            if not (non_abstract_only and cls._IsAbstract):
                subclses.add(cls)
            all_clses.extend(cls.__subclasses__())
        return tuple(subclses)
    
    @classproperty  # type: ignore
    def _IsAbstract(cls):
        '''Test if this node is abstract.'''
        if cls.IsAbstract is not None:
            return cls.IsAbstract
        return isabstract(cls)
    
    @classproperty  # type: ignore
    def _Category(cls):
        '''Return the category for registration.'''
        cls_category = cls.Category
        if cls_category:
            super_cls = cls.__bases__[0]
            if super_cls.Category is not None:
                super_cls_cat = super_cls.Category
                if super_cls_cat.endswith('/'):
                    super_cls_cat = super_cls_cat[:-1]
                if cls_category.startswith('/'):
                    cls_category = cls_category[1:]
                cls_category = f"{super_cls.Category}/{cls_category}" 
        return cls_category
    
    @classproperty  # type: ignore
    def _ReadableName(cls)->str:
        '''Return the real user friendly class name for registration.'''
        name = cls.Name
        if not name:
            name = ""
            cls_name = cls.__qualname__
            if cls_name.endswith('Node') and len(name) > 4:
                cls_name = cls_name[:-4]

            name_chunks = cls_name.split('_')   # split by underscore first
            for i in range(len(name_chunks)):
                # find camel case, e.g. "MyNode" -> "My Node"
                if camel_chunks := re.findall(r'[A-Z][a-z]*', name_chunks[i]):
                    name += ' '.join(camel_chunks) + ' '
                else:
                    name += name_chunks[i] + ' '
        return name
    
    @classmethod
    def _HiddenParamNameMapping(cls, sig: Signature):
        param_name_map = {}
        for key, param in sig.parameters.items():
            param_type = get_input_param_type(key, param)
            if param_type == 'hidden':
                param_name_map[_get_proper_param_name(key)] = key
        return param_name_map
    
    @classmethod
    def _InputTypes(cls, sig: Signature):
        '''Real input types. If not defined, it will be determined by `__call__` method.'''
        
        require_types = {}
        optional_types = {}
        hidden_types = {}
        
        for i, (key, param) in enumerate(sig.parameters.items()):
            if i == 0:
                continue    # skip `self`
            param_type = get_input_param_type(key, param)
            param_info = get_type_info_for_comfyUI(param)
            if param_type == 'required':
                require_types[_get_proper_param_name(key)] = param_info
            elif param_type == 'optional':
                optional_types[_get_proper_param_name(key)] = param_info
            elif param_type == 'hidden':
                hidden_types[_get_proper_param_name(key)] = param_info
                
        ret_dict = {}
        if require_types:
            ret_dict['required'] = require_types
        if optional_types:
            ret_dict['optional'] = optional_types
        if hidden_types:
            ret_dict['hidden'] = hidden_types
        return ret_dict

    @classmethod
    def _ReturnTypes(cls, sig: Signature):
        '''Real return types. If not defined, it will be determined by `__call__` method.'''
        return get_return_type_and_names_for_comfyUI(sig.return_annotation)
    
    @classproperty  # type: ignore
    def _RealComfyUINode(cls)->type:
        '''Return the real registration node class for ComfyUI'''
        if cls._IsAbstract:
            raise TypeError(f'Abstract node {cls} cannot be registered. `_RealComfyUINode` should not be called on abstract nodes.')
        
        if hasattr(cls, '__RealComfyUINode__'):
            return cls.__RealComfyUINode__
        
        newcls = type(cls.__qualname__, (), {}) # create a new class
        
        category = cls._Category
        if category:
            setattr(newcls, 'CATEGORY', category)
        
        calling_sig = signature(cls.__call__)
        params = calling_sig.parameters
        
        enum_params = {k: v for k, v in params.items() if (type(v.annotation) == type and issubclass(v.annotation, Enum))} 
        latent_params = {k: v for k, v in params.items() if (type(v.annotation) == type and issubclass(v.annotation, LATENT))}
        
        ret = cls._ReturnTypes(calling_sig)
        ui_return_type: UIType = None
        return_types, return_names = None, None
        
        if isinstance(ret, UIType):
            ui_return_type = ret
            setattr(newcls, 'RETURN_TYPES', tuple())    # for ui output types, RETURN_TYPES should be empty
            setattr(newcls, 'OUTPUT_NODE', True)
        else:
            return_types, return_names = ret
            setattr(newcls, 'RETURN_TYPES', return_types)
            if return_names:
                setattr(newcls, 'RETURN_NAMES', return_names)
            setattr(newcls, 'OUTPUT_NODE', False)
        
        hidden_param_name_map = cls._HiddenParamNameMapping(calling_sig)
        
        def proper_input(*args, **kwargs):            
            for k, v in hidden_param_name_map.items():
                if k in kwargs:
                    kwargs[v] = kwargs.pop(k)

            if enum_params or latent_params: # find the real enum value
                packed_params = _pack_param(calling_sig, args, kwargs)
                for k, v in enum_params.items():
                    input_enum_name = packed_params[k]
                    for enum_name, enum_val in v.annotation.__members__.items():
                        if enum_name == input_enum_name:
                            packed_params[k] = enum_val
                            break
                for latent_name, latent_val in latent_params.items():
                    packed_params[latent_name] = LATENT(latent_val)
                
                return tuple(), packed_params
            else:
                return args, kwargs
        
        def real_calling(ins, *args, **kwargs):
            if not hasattr(ins, '__real_node_instance__'):
                setattr(ins, '__real_node_instance__', cls())

            proper_args, proper_kwargs = proper_input(*args, **kwargs)
            if proper_args:
                result = ins.__real_node_instance__.__call__(*proper_args, **proper_kwargs)
            else:
                result = ins.__real_node_instance__.__call__(**proper_kwargs)

            if ui_return_type is not None:
                if isinstance(result, Sequence):
                    result = result[0]
                return ui_return_type.to_dict(result)
            else:
                if not isinstance(result, Sequence) and len(return_types) == 1:
                    result = (result,)
                else:
                    result = tuple(result)
            return _return_proper_value(result)
        
        setattr(newcls, 'real_calling', real_calling)
        setattr(newcls, 'FUNCTION', 'real_calling')
        
        @classmethod
        def INPUT_TYPES(s):
            return cls._InputTypes(calling_sig)
        setattr(newcls, 'INPUT_TYPES', INPUT_TYPES)
        
        @classmethod
        def IS_CHANGED(s, *args, **kwargs):
            proper_args, proper_kws = proper_input(*args, **kwargs)
            if proper_args:
                cls.OnInputChanged(*proper_args, **proper_kws)
            else:
                cls.OnInputChanged(**proper_kws)
        setattr(newcls, 'IS_CHANGED', IS_CHANGED)
        
        setattr(cls, '__RealComfyUINode__', newcls)
        return newcls
    # endregion
    
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        '''
        Each node should implement `__call__` method to define its behavior.
        You must define input/return annotation for this method, so as to register the node properly.
        
        Note:
            Although it supports Enum as input param, it is recommended to use `Literal` first as it has lower time complexity.
        
        Example:
        ```python
        class MyNode1(NodeBase):
            def __call__(self, 
                        x:INT(min=0, max=10), # you can use the special annotation for constraints. See `INT`, `FLOAT`, `STRING`
                        y:int = 1,            # set default value using `=`
                        z:float = None,       # set `optional` param using `= None`
                        _w: str = 'hello'     # set `hidden` param using `_`
                        )->float:
                return x + y + (z or 0.0)
        
        class MyNode2(NodeBase):
            def __call__(self, 
                        x: Literal['a', 'b'],     # you can use the `Literal`/Enum for input type
                        y: LATENT,               # you can use `LATENT` for input type
                        )... 
        ```
        '''

    @classmethod
    def OnInputChanged(cls, *args, **kwargs):
        '''
        This method will be called when the input of this node is changed.
        You can override this method to do some extra operations when the input is changed.
        
        The arguments must be the same as `__call__` method.
        '''
        pass

class StableRendererNodeBase(NodeBase):
    '''For inheritance category of stable-renderer nodes.'''
    
    Category = 'stable-renderer'



__all__ = ['NodeBase', 'StableRendererNodeBase']

