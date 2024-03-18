'''
Base class for all nodes.
All sub classes will be registered automatically.
'''

import re
from inspect import isabstract, signature, Parameter, Signature
from abc import ABC, abstractmethod
from typing import (ClassVar, Optional, Literal, Any, Tuple, Annotated, TypeAlias, Union, Dict, 
                    get_args, overload, TypeVar, Generic, List)
from enum import Enum
from collections import OrderedDict
from common_utils.type_utils import get_origin, get_cls_name
from comfyUI.types import *


NT = TypeVar('NT', bound='NodeBase')

class SRComfyNode(ComfyUINode, Generic[NT]):
    '''Type hints for advanced comfyUI nodes defined by `NodeBase`.'''

    CATEGORY: Optional[str]
    '''The category of this node. This is for grouping nodes in the UI.'''
    DESCRIPTION: str
    '''The description of this node. This is for registration.'''
    RETURN_TYPES: Tuple[str, ...]
    '''The return types of this node. This is for registration.'''
    
    __real_node_instance__: NT
    '''The real instance of the node class.'''

__all__ = ['SRComfyNode']

def _get_input_param_type(key: str, param: Parameter)->Literal['required', 'optional', 'hidden']:
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

def _pack_param(sig: Signature, args, kwargs)-> OrderedDict[str, Any]:
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
            raise TypeError(f'Unexpected keyword argument: {k}')
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
                    raise TypeError(f'Missing required positional argument: {name}')
                packed_params[name] = param.default
    
    if args_copy:
        raise TypeError(f'Unexpected positional arguments: {args_copy}')

    if var_args_field_name:
        packed_params[var_args_field_name] = tuple(packed_params[var_args_field_name])
        
    return packed_params

def _return_proper_value(values: Tuple[Any, ...])->List[Any]:
    '''Proper the return types to be suitable for ComfyUI.'''
    vals = []
    for val in values:
        if isinstance(val, Enum):
            vals.append(val.name)
        else:
            vals.append(val)
    return vals   


class _NodeField:
    '''Data class for containing the field info of a node.'''
    
    origin_param_name: Optional[str]
    '''The original name of the param. `None` is only for return types.'''
    param_name: Optional[str]
    '''The name of the param. `None` is only for return types.'''
    param: AnnotatedParam
    '''infos'''
    param_type: Optional[Literal['required', 'optional', 'hidden']]
    '''The type of the param. `None` is only for return types.'''
    
    @overload
    def __init__(self, param: Parameter): '''input field info'''
    
    @overload
    def __init__(self, param: Union[type, TypeAlias]): '''return field info'''
    
    def __init__(self, param: Union[type, TypeAlias, Parameter]):
        if isinstance(param, Parameter):    # input field info
            self.origin_param_name = param.name
            self.param_name = _get_proper_param_name(self.origin_param_name)
            
            anno = param.annotation
            origin = get_origin(anno)
            
            if origin:
                if origin == Annotated:
                    args = get_args(anno)
                    if len(args) != 2:
                        raise TypeError(f'Annotated should have 2 args, but got {args}')
                    second_param_cls_name = get_cls_name(args[1])
                    if second_param_cls_name != 'AnnotatedParam':   # due to some import problem, must be checked by class name
                        raise TypeError(f'Annotated should have AnnotatedParam as the second arg, but got {args[1]}')
                    self.param = args[1]
                elif origin in (Literal, Any):
                    annotated_param = AnnotatedParam(origin_type=anno)
                    self.param = annotated_param
                else:
                    raise TypeError(f'Unsupported origin type {origin}')
            else:   # type annotation
                annotated_param = AnnotatedParam(origin_type=anno)
                self.param = annotated_param
                
            if param.default != Parameter.empty:
                self.param.default = param.default

            self.param_type = _get_input_param_type(self.origin_param_name, param)
        else:   # return field info
            self.origin_param_name = None
            self.param_name = None
            self.param_type = None
            
            if param == Parameter.empty:
                param = Any # type: ignore
            origin = get_origin(param)
            if origin:
                if origin == Annotated:
                    args = get_args(param)
                    if len(args) != 2 or not isinstance(args[1], (AnnotatedParam, str)):
                        raise TypeError(f'Annotated should have 2 args, but got {args}')
                    if isinstance(args[1], str):    # e.g. Annotated[int, 'MyInt'] means you have a return value named 'MyInt'
                        self.origin_param_name = args[1]
                        self.param_name = _get_proper_param_name(self.origin_param_name)
                        self.param = AnnotatedParam(origin_type=args[0])
                    else:
                        self.param = args[1]
                elif origin in (Literal, Any):
                    self.param = AnnotatedParam(origin_type=param)
                else:
                    raise TypeError(f'Unsupported origin type {origin}')
            else:
                self.param = AnnotatedParam(origin_type=param)

class NodeBase(ABC):
    '''
    This is an advance base class for customizing nodes.
    As ComfyUI is lack of documents on customization, inheritance from this class helps you to define nodes easily and correctly.
    
    '''
    
    IsAbstract: ClassVar[Optional[bool]] = None
    '''
    Define whether this node is abstract or not. If None, it will be determined automatically.
    Abstract nodes will not be registered.
    '''
    Category: ClassVar[Optional[str]] = None
    '''The category of this node. This is for grouping nodes in the UI.'''
    Name: ClassVar[Optional[str]] = None
    '''The user friendly name of this node. This is for registration. If not defined, the class name(split by camel case & underscore) will be used.'''
    Description: ClassVar[Optional[str]] = None
    '''
    The description of this node. 
    This is for registration. If not defined, it will try using __doc__ of the class.
    '''
    
    _CallSig: ClassVar[Signature]
    '''The signature of `__call__` method. This is for registration.'''
    _InputFields: ClassVar[Dict[str, _NodeField]]
    '''
    The input fields of this node. This is for registration.
    {origin_param_name: _NodeField, ...}
    '''
    _ReturnFields: ClassVar[Tuple[Tuple[Union[str, None], _NodeField], ...]]
    '''The return fields of this node. This is for registration.'''
    _ParamNameMap: ClassVar[Dict[str, str]]
    '''Dict for mapping the shown param name to the original param name(for hidden params)'''
    _ReadableName: ClassVar[str]
    '''The real user friendly class name for registration.'''
    _Description: ClassVar[str]
    '''The real description for registration.'''
    _Category: ClassVar[Optional[str]]
    '''The real category for registration.'''
    
    _HasIsChangedMethod: ClassVar[bool]
    '''Whether this node has override `IsChanged` method or not. This is for registration.'''
    _HasValidateInputMethod: ClassVar[bool]
    '''Whether this node has override `ValidateInput` method or not. This is for registration.'''
    
    _IsAbstract: ClassVar[bool]
    '''Real value for `IsAbstract`.'''
    
    _RealComfyUINode: ClassVar[type] = None     # type: ignore
    '''The real registration node class for ComfyUI'''
    
        
    @classmethod
    def _InitFields(cls):
        cls._InputFields = {param_name: _NodeField(param) for param_name, param in cls._CallSig.parameters.items()}
        
        return_anno = cls._CallSig.return_annotation
        if return_anno == Parameter.empty:
            return_anno = Any
        origin = get_origin(return_anno)
        return_field_names = []
        return_fields = []
        if origin:  # special types, e.g. Annotated, Literal
            if origin in (Annotated, Literal, Any):
                field = _NodeField(return_anno)
                return_field_names.append(field.param_name) # can append `None`
                return_fields.append(field)
            elif origin in (Tuple, tuple):   # multiple return values
                args = get_args(origin)
                for arg in args:
                    field = _NodeField(arg)
                    return_field_names.append(field.param_name)
                    return_fields.append(field)
            else:
                raise TypeError(f'Unexpected return annotation: {return_anno}')
        else:
            field = _NodeField(return_anno)
            return_field_names.append(field.param_name) # can append `None`
            return_fields.append(field)
        
        no_name_count = return_field_names.count(None)
        if no_name_count not in (0, len(return_field_names)):   # all or none should have names
             raise TypeError(f'If you have named any of the return value, all of the return values should be named')
        cls._ReturnFields = tuple(zip(return_field_names, return_fields))   # named return values
    
    @classmethod
    def _InitCategory(cls):
        '''Return the real category for registration.'''
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
        cls._Category = cls_category
    
    @classmethod
    def _InitReadableName(cls):
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
        cls._ReadableName = name
    
    @classmethod
    def _InitDescription(cls):
        '''Return the real description for registration.'''
        desc = cls.Description
        if not desc:
            if hasattr(cls, '__doc__'):
                desc = cls.__doc__ or ''
            else:
                desc = ''
        cls._Description = desc
    
    @classmethod
    def _InitRealComfyUINode(cls):
        '''Return the real registration node class for ComfyUI'''
        if cls._IsAbstract:
            raise TypeError(f'Abstract node {cls} cannot be registered. `_RealComfyUINode` should not be called on abstract nodes.')
        
        newcls_name = cls.__qualname__
        if newcls_name.endswith('Node') and len(newcls_name) > 4:
            newcls_name = newcls_name[:-4]
        
        newcls: SRComfyNode = type(newcls_name, (), {}) # type: ignore
        
        def newcls_init(newcls_self):
            newcls_self.__real_node_instance__ = cls()  # create the real instance of this node cls when the newcls instance is created
        setattr(newcls, '__init__', newcls_init)
        
        category = cls._Category
        if category:
            setattr(newcls, 'CATEGORY', category)
        
        setattr(newcls, 'DESCRIPTION', cls._Description)

        has_ui_ret = any((field.param.tags and 'ui' in field.param.tags) for (_, field) in cls._ReturnFields)
        setattr(newcls, 'OUTPUT_NODE', has_ui_ret)    # `OUTPUT_NODE` is a flag for ComfyUI to shown on UI
        
        return_names = [name for name, _ in cls._ReturnFields if name]
        return_types = [field.param._comfy_type_definition for _, field in cls._ReturnFields]   # not sure ComfyUI support COMBO return? maybe error in future
        setattr(newcls, 'RETURN_TYPES', tuple(return_types))
        
        if return_names and return_names[0]:    # just check the first name is not None is enough(should be all or none)
            setattr(newcls, 'RETURN_NAMES', return_names)
        
        output_is_list_types = []
        for (_, field) in cls._ReturnFields:
            if field.param.origin_type in (list, List):
                output_is_list_types.append(True)
            else:
                output_is_list_types.append(False)
        if any(output_is_list_types):
            # for comfyUI to know whether the output is a list
            setattr(newcls, 'OUTPUT_IS_LIST', tuple(output_is_list_types))  
            
        enum_fields = {k: v for k, v in cls._InputFields.items() if (type(v.param.origin_type) == type and issubclass(v.param.origin_type, Enum))}
        latent_fields = {k: v for k, v in cls._InputFields.items() if v.param.origin_type == LATENT}
        
        hidden_params = {}
        for proper_name, real_name in cls._ParamNameMap.items():
            if cls._InputFields[real_name].param_type == 'hidden':
                hidden_params[proper_name] = real_name
        
        def proper_input(*args, **kwargs):            
            for k, v in hidden_params.items():
                if k in kwargs:
                    kwargs[v] = kwargs.pop(k)   # change the shown name to the real name

            if enum_fields or latent_fields: # find the real enum value
                packed_params = _pack_param(cls._CallSig, args, kwargs)
                for _, field in enum_fields.items():
                    input_enum_name = packed_params[k]
                    for enum_name, enum_val in field.param.origin_type.__members__.items(): # type: ignore
                        if enum_name == input_enum_name:
                            packed_params[k] = enum_val
                            break
                for _, field in latent_fields.items():
                    if not isinstance(packed_params[k], LATENT):
                        packed_params[k] = LATENT(packed_params[k])
                return tuple(), packed_params
            else:
                return args, kwargs
        
        def real_calling(ins, *args, **kwargs):
            proper_args, proper_kwargs = proper_input(*args, **kwargs)
            result = ins.__real_node_instance__.__call__(*proper_args, **proper_kwargs)

            if len(return_types) == 1 and not isinstance(result, tuple):
                result = (result,)
            else:
                result = tuple(result)
            vals = _return_proper_value(result)
            if has_ui_ret:
                return cls._MakeUIRetDict(vals)
            else:
                return vals
            
        setattr(newcls, '_real_calling', real_calling)
        setattr(newcls, 'FUNCTION', '_real_calling') # comfyUI will call `real_calling` method
        
        @classmethod
        def INPUT_TYPES(s):
            return cls._InputTypesForComfy()
        setattr(newcls, 'INPUT_TYPES', INPUT_TYPES)
        
        lazy_inputs = []
        reduce_inputs = []
        for name, field in cls._InputFields.items():
            if field.param.tags is not None:
                if 'lazy' in field.param.tags:
                    lazy_inputs.append(name)
                if 'reduce' in field.param.tags:
                    reduce_inputs.append(name)
        if lazy_inputs:
            setattr(newcls, 'LAZY_INPUTS', tuple(lazy_inputs))
        if reduce_inputs:
            setattr(newcls, 'REDUCE_INPUTS', tuple(reduce_inputs))
        
        if cls._HasIsChangedMethod:
            def IS_CHANGED(self, *args, **kwargs):
                proper_args, proper_kws = proper_input(*args, **kwargs)
                return cls.IsChanged(*proper_args, **proper_kws)
            setattr(newcls, 'IS_CHANGED', IS_CHANGED)
        
        if cls._HasValidateInputMethod:
            def VALIDATE_INPUTS(self, *args, **kwargs):
                proper_args, proper_kws = proper_input(*args, **kwargs)
                return cls.ValidateInput(*proper_args, **proper_kws)
            setattr(newcls, 'VALIDATE_INPUTS', VALIDATE_INPUTS)
            
        cls._RealComfyUINode = newcls
    
    def __init_subclass__(cls):
        if cls.IsAbstract is not None:
            cls._IsAbstract = cls.IsAbstract
        else:
            cls._IsAbstract = isabstract(cls)
        
        cls._InitCategory()
        
        if not cls._IsAbstract:
            cls._CallSig = signature(cls.__call__)
            cls._InitReadableName()
            cls._InitDescription()
            cls._InitFields()
            
            cls._ParamNameMap = {}
            for param_name in cls._InputFields:
                cls._ParamNameMap[_get_proper_param_name(param_name)] = param_name
            
            cls._HasIsChangedMethod = cls.IsChanged != NodeBase.IsChanged
            if cls._HasIsChangedMethod:
                onchanged_method_sig = signature(cls.IsChanged)
                for i, (_, param) in enumerate(onchanged_method_sig.parameters.items()):
                    if i == 0:
                        continue    # skip `self`
                    if param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                        break   # has args except `*args` and `**kwargs`, should be overridden
                    if i == len(onchanged_method_sig.parameters) - 1:
                        cls._HasIsChangedMethod = False # only `*args` and `**kwargs`, not overridden
            
            if cls._HasIsChangedMethod: # TODO: more strict check
                if len(onchanged_method_sig.parameters) != len(cls._CallSig.parameters):
                    raise TypeError(f'`IsChanged` method should have the same signature as `__call__` method. {cls.__name__}')
            
            cls._HasValidateInputMethod = cls.ValidateInput != NodeBase.ValidateInput
            if cls._HasValidateInputMethod:
                onvalidate_method_sig = signature(cls.ValidateInput)
                for i, (_, param) in enumerate(onvalidate_method_sig.parameters.items()):
                    if i == 0:
                        continue    # skip `cls`
                    if param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                        break
                    if i == len(onvalidate_method_sig.parameters) - 1:
                        cls._HasValidateInputMethod = False
            
            if cls._HasValidateInputMethod:
                if len(onvalidate_method_sig.parameters) != len(cls._CallSig.parameters):
                    raise TypeError(f'`ValidateInput` method should have the same signature as `__call__` method. {cls.__name__}')
            
            cls._InitRealComfyUINode()
    
    # region internal use
    @classmethod
    def _MakeUIRetDict(cls, values: List[Any]):
        '''return a dictionary for comfyUI to show the return values labeled with "UI"'''
        ui_rets = {}
        animated = []
        for i, (name, field) in enumerate(cls._ReturnFields):
            if field.param.tags is not None:
                if 'ui' in field.param.tags:
                    ui_rets[name] = values[i]
                
                if 'animated' in field.param.tags:
                    animated.append(True)
                else:
                    animated.append(False)
        if any(animated):
            ui_rets['animated'] = tuple(animated)   # not wrong, comfyUI put `animated` in the `ui` dict
        ret = {'ui': ui_rets, 'result': tuple(values)}
        return ret
        
    @staticmethod
    def _AllSubclasses(non_abstract_only: bool = True):
        '''return all subclasses of `NodeBase`.'''
        all_clses = list(NodeBase.__subclasses__())
        subclses = set()
        while all_clses:
            cls = all_clses.pop()
            if not (non_abstract_only and cls._IsAbstract):
                subclses.add(cls)
            all_clses.extend(cls.__subclasses__())
        return tuple(subclses)
    
    @classmethod
    def _InputTypesForComfy(cls):
        '''Real input type dictionary for registration.'''
        
        require_types = {}
        optional_types = {}
        hidden_types = {}
        
        for i, (key, field) in enumerate(cls._InputFields.items()):
            if i == 0:
                continue    # skip `self`
            param_type = field.param_type
            param_info = field.param._tuple
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

    def IsChanged(self, *args, **kwargs)->Any:
        '''
        Define this method to decide whether the node should be executed again.
        This method should return a value for comparing the uniqueness of the input.
        
        e.g. return the hash of a file to check whether the file is changed.
        
        If not defined, the node will be executed every time when the input is changed.
        '''
    
    @classmethod
    def ValidateInput(cls, *args, **kwargs)->bool:
        '''
        You can define this method to validate the node input is valid. Note that this is a classmethod
        (I think thats a bad idea actually, should be changed into instance method)
        
        return a boolean value to indicate whether the input is valid.
        '''


class StableRendererNodeBase(NodeBase):
    '''For inheritance category of stable-renderer nodes.'''
    
    Category = 'stable-renderer'



__all__.extend(['NodeBase', 'StableRendererNodeBase'])
