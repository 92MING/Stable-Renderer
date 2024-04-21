'''
Base class for all nodes.
All sub classes will be registered automatically.
'''
import re

from inspect import isabstract, signature, Parameter, Signature
from abc import abstractmethod
from typing import (Type, _ProtocolMeta, ClassVar, Optional, Any, Tuple, Union, Dict, get_args, List, runtime_checkable, Protocol,
                    TypeAlias, Literal)
from collections import OrderedDict

from common_utils.base_clses import CrossModuleABC
from common_utils.debug_utils import ComfyUILogger
from common_utils.type_utils import get_origin, is_empty_method

from .basic import *

class _ComfyUINodeMeta(_ProtocolMeta):
    def __instancecheck__(cls, instance: Any) -> bool:
        if hasattr(instance, '__IS_COMFYUI_NODE__') and instance.__IS_COMFYUI_NODE__:
            return True
        return False
    
    def __subclasscheck__(cls, subclass: Type) -> bool:
        return hasattr(subclass, '__IS_COMFYUI_NODE__') and subclass.__IS_COMFYUI_NODE__
        
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
    '''real node instance after creation. This is only available for advanced comfyUI nodes which inherit from `NodeBase`.'''
    
    ID: str
    '''The unique id of the node. This will be assigned in runtime.'''
    # endregion
    
    FUNCTION: str
    '''the target function name of the node. If not define, will try to use the only function it have.'''
    DESCRIPTION: str
    '''the description of the node'''
    CATEGORY: str
    '''the category of the node. For searching'''
    NAMESPACE: Union[str, None] = None
    '''the namespace of the node. It is used to avoid name conflicts.'''
    
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
    def INPUT_TYPES(cls) -> Dict[NodeInputParamType, Dict[str, Tuple[Union[str, type], dict]]]: # type: ignore
        '''
        All nodes for comfyUI should have this class method to define the input types.
        
        {input_type: {param_name: (param_type, param_info_dict), ...}, ...}
        '''
    
    RETURN_TYPES: Tuple[str, ...]
    '''All return type names of the node.'''
    RETURN_NAMES: Tuple[str, ...]
    '''Names for each return values'''
    
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs)->bool:     # type: ignore
        '''If you have defined this method, it will be called to validate the input values.'''    
    
    def IS_CHANGED(self, *args, **kwargs): 
        '''
        This method will be called when the input value is changed. 
        It should have the same signature as the function which you have defined to called.
        '''
    
    def ON_DESTROY(self):
        '''This method will be called when the node is destroyed.'''
    
    
__all__ = ['NodeInputParamType', 'ComfyUINode']


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
    kwargs_copy = dict(kwargs)
    for name, param in parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:  # *args param
            for arg in args_copy:
                packed_params[name].append(arg) # already created a list in the first step, i.e. name == self.var_args_field_name
                args_copy = args_copy[1:]
        elif param.kind == Parameter.VAR_KEYWORD:   # **kwargs param
            for (k, v) in tuple(packed_params.items()):
                packed_params[name][k] = v  # already created a dict in the first step, i.e. name == self.var_kwargs_field_name
        else:   # normal param
            if name in kwargs:
                packed_params[name] = kwargs_copy.pop(name)
            elif len(args_copy)>0:
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

class NodeBase(CrossModuleABC):
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
    NameSpace: ClassVar[Optional[str]] = None
    '''namespace for avoiding name conflict. This is for registration & searching.'''
    Description: ClassVar[Optional[str]] = None
    '''
    The description of this node. 
    This is for registration. If not defined, it will try using __doc__ of the class.
    '''
    
    _CallSig: ClassVar[Signature]
    '''The signature of `__call__` method. This is for registration.'''
    
    _InputFields: ClassVar[Dict[str, AnnotatedParam]]
    '''
    The input fields of this node. This is for registration.
    {origin_param_name: _NodeField, ...}
    '''
    _ReturnFields: ClassVar[Tuple[Tuple[Union[str, None], AnnotatedParam], ...]]
    '''The return fields of this node. This is for registration.'''
    _LazyInputs: ClassVar[Tuple[str, ...]]
    '''input param names that are lazy evaluated. This is for registration.'''
    _ListInputs: ClassVar[Tuple[str, ...]]
    '''input param names that are list. For internal use only.'''
    _ParamNameMap: ClassVar[Dict[str, str]]
    '''Dict for mapping the shown param name to the original param name(for hidden params)'''
    _ComfyInputParamDict: ClassVar[dict]
    '''The real input type dictionary for registration in ComfyUI.'''
    _ComfyOutputTypes: ClassVar[Tuple[str, ...]]
    '''The real tuple containing output type names for registration in ComfyUI.'''
    _ComfyOutputNames: ClassVar[Tuple[Union[str, None], ...]]
    '''The real tuple containing output names for registration in ComfyUI.'''
    
    _ReadableName: ClassVar[str]
    '''The real user friendly class name for registration.'''
    _RealClsName: ClassVar[str]
    '''The real class name for registration.'''
    _Description: ClassVar[Optional[str]]
    '''The real description for registration.'''
    _Category: ClassVar[Optional[str]]
    '''The real category for registration.'''
    
    _HasIsChangedMethod: ClassVar[bool]
    '''Whether this node has override `IsChanged` method or not. This is for registration.'''
    _HasValidateInputMethod: ClassVar[bool]
    '''Whether this node has override `ValidateInput` method or not. This is for registration.'''
    _HasUIReturn: ClassVar[bool]
    '''Whether this node has UI return value or not. This is for registration.'''
    _HasListInput: ClassVar[bool]
    '''Whether this node has list input or not. This is for registration.'''
    _ListTypeReturns: ClassVar[Tuple[bool, ...]]
    '''Which return value is a list or not. This is for registration.'''
    _IsAbstract: ClassVar[bool]
    '''Real value for `IsAbstract`.'''
    
    _RealComfyUINodeCls: ClassVar[Type[ComfyUINode]]
    '''The real registration node class for ComfyUI'''
    
    @classmethod
    def _InitFields(cls):
        # init input fields
        cls._InputFields = {param_name: AnnotatedParam(param) for param_name, param in cls._CallSig.parameters.items()}
        
        # init return fields
        return_anno = cls._CallSig.return_annotation
        if return_anno == Parameter.empty:
            return_anno = Any
        origin = get_origin(return_anno)
        
        return_field_names: List[Optional[str]] = []
        return_fields: List[AnnotatedParam] = []
        
        if origin:  # special types, e.g. Annotated, Literal

            if origin == tuple:   # possible: multiple return values/ 1 named return value / tuple[type, ...](means array)
                return_types = get_args(return_anno)
                if len(return_types) == 1:
                    raise TypeError(f'Unexpected return annotation: {return_anno}')
                
                if len(return_types) == 2 and return_types[1] == Ellipsis:  # treat as array return values
                    real_type = return_types[0]
                    if isinstance(real_type, AnnotatedParam):
                        real_type.tags.add('list')
                        return_fields.append(real_type)
                        return_field_names.append(real_type.param_name or "")
                    else:
                        return_fields.append(AnnotatedParam(real_type, tags=['list']))
                        return_field_names.append("")
                
                else:   # several return values
                    for ret_type in return_types:
                        return_fields.append(AnnotatedParam(ret_type))
                        return_field_names.append(return_fields[-1].param_name or "")
            else:
                return_fields.append(AnnotatedParam(return_anno))    # type: ignore
                return_field_names.append(return_fields[-1].param_name or "")
        else:
            return_fields.append(AnnotatedParam(return_anno))    # type: ignore
            return_field_names.append(return_fields[-1].param_name or "")
        
        cls._HasUIReturn = any(('ui' in field.tags) for field in return_fields)
        cls._ListInputs = tuple([name for name, field in cls._InputFields.items() if 'list' in field.tags])
        cls._HasListInput = bool(cls._ListInputs)
        cls._ReturnFields = tuple(zip(return_field_names, return_fields))   # named return values
        cls._ListTypeReturns = tuple([('list' in t.tags) for _, t in cls._ReturnFields])

    @classmethod
    def _InitBasicInfo(cls):
        '''init cls name, readable name, category, description for registration.'''
        # readable name
        readable_name = cls.__qualname__
        if readable_name.lower().endswith('node') and len(readable_name) > 4:
            readable_name = readable_name[:-4]
        if readable_name.lower().startswith('stablerenderer') and len(readable_name) > 13:
            readable_name = readable_name[13:]

        readable_name_chunks = readable_name.split('_')   # split by underscore first
        readable_name = ""
        for chunk in readable_name_chunks:
            # find camel case, e.g. "MyNode" -> "My Node"
            chunked_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', chunk)
            readable_name += chunked_name
            readable_name += " "
        cls._ReadableName = readable_name.strip()
        
        # class name
        real_clsname = cls.__qualname__
        if real_clsname.lower().endswith('node') and len(real_clsname) > 4:
            real_clsname = real_clsname[:-4]
        if real_clsname.lower().startswith('stablerenderer') and len(real_clsname) > 13:
            real_clsname = real_clsname[13:]
        cls._RealClsName = real_clsname
        
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
        
        # description
        desc = cls.Description
        if not desc:
            if hasattr(cls, '__doc__'):
                desc = cls.__doc__ or ''
            elif hasattr(cls.__call__, '__doc__'):
                desc = cls.__call__.__doc__ or ''
        cls._Description = desc  # type: ignore
    
    @classmethod
    def _InitTypesForComfy(cls):
        '''Real input type dictionary for registration.'''
        # input types
        require_types = {}
        optional_types = {}
        hidden_types = {}
        
        for i, (_, field) in enumerate(cls._InputFields.items()):
            if i == 0:
                continue    # skip `self`
            param_type = field.param_type
            param_info = field._comfyUI_definition
            if param_type == 'required':
                require_types[field.proper_param_name] = param_info
            elif param_type == 'optional':
                optional_types[field.proper_param_name] = param_info
            elif param_type == 'hidden':
                hidden_types[field.proper_param_name] = param_info
        
        input_param_dict = {}
        if require_types:
            input_param_dict['required'] = require_types
        if optional_types:
            input_param_dict['optional'] = optional_types
        if hidden_types:
            input_param_dict['hidden'] = hidden_types
        cls._ComfyInputParamDict = input_param_dict    
        
        # return types
        return_names = [name for name, _ in cls._ReturnFields]
        return_types = [field._comfy_type for (_, field) in cls._ReturnFields]   # not sure ComfyUI support COMBO return? maybe error in future
        cls._ComfyOutputTypes = tuple(return_types) # type: ignore
        cls._ComfyOutputNames = tuple(return_names)
        
        # lazy inputs
        lazy_inputs = [name for name, field in cls._InputFields.items() if 'lazy' in field.tags]
        cls._LazyInputs = tuple(lazy_inputs)
        
    @classmethod
    def _InitRealComfyUINode(cls):
        '''Return the real registration node class for ComfyUI'''
        if cls._IsAbstract:
            raise TypeError(f'Abstract node {cls} cannot be registered. `_RealComfyUINode` should not be called on abstract nodes.')
        
        newcls: Type[ComfyUINode] = type(cls._RealClsName, (), {})  # type: ignore
        
        setattr(newcls, '__IS_COMFYUI_NODE__', True)
        setattr(newcls, '__IS_ADVANCED_COMFYUI_NODE__', True)
        setattr(newcls, '__ADVANCED_NODE_CLASS__', cls)
        
        def newcls_init(newcls_self):
            newcls_self.__real_node_instance__ = cls()  # create the real instance of this node cls when the newcls instance is created
        setattr(newcls, '__init__', newcls_init)
        
        if cls._Category:
            setattr(newcls, 'CATEGORY', cls._Category)
        if cls._Description:
            setattr(newcls, 'DESCRIPTION', cls._Description)
        if cls._HasUIReturn:
            setattr(newcls, 'OUTPUT_NODE', cls._HasUIReturn)    # `OUTPUT_NODE` is a flag for ComfyUI to shown on UI
        setattr(newcls, 'NAMESPACE', cls.NameSpace if hasattr(cls, 'NameSpace') else "")
        
        setattr(newcls, 'RETURN_TYPES', tuple(cls._ComfyOutputTypes))
        
        if any(cls._ComfyOutputNames):    # just check the first name is not None is enough(should be all or none)
            setattr(newcls, 'RETURN_NAMES', cls._ComfyOutputNames)
        
        if cls._HasListInput:
            setattr(newcls, 'INPUT_IS_LIST', True)
        
        if any(cls._ListTypeReturns):
            # for comfyUI to know whether the output is a list
            setattr(newcls, 'OUTPUT_IS_LIST', cls._ListTypeReturns)  
        
        setattr(newcls, 'LAZY_INPUTS', cls._LazyInputs)
        
        hidden_params = {}
        for proper_name, real_name in cls._ParamNameMap.items():
            if cls._InputFields[real_name].param_type == 'hidden':
                hidden_params[proper_name] = real_name
        
        def proper_input(ins, *args, **kwargs):            
            for k, v in hidden_params.items():
                if k in kwargs:
                    kwargs[v] = kwargs.pop(k)   # change the shown name to the real name

            packed_params = _pack_param(cls._CallSig, (ins, *args), kwargs)
            for param_name, val in packed_params.items():
                packed_params[param_name] = cls._InputFields[param_name].format_value(val)
            return packed_params
        
        def real_calling(ins, *args, **kwargs):
            proper_kwargs = proper_input(ins, *args, **kwargs)
            result = cls.__call__(**proper_kwargs)  # 'self' is included in the `proper_kwargs`

            if len(cls._ComfyOutputTypes) == 1 and not isinstance(result, tuple):
                result = (result,)
            if cls._HasUIReturn:
                return cls._MakeUIRetDict(result)    # type: ignore
            else:
                return result
            
        setattr(newcls, '_real_calling', real_calling)
        setattr(newcls, 'FUNCTION', '_real_calling') # comfyUI will call `real_calling` method
        
        @classmethod
        def INPUT_TYPES(s):
            return cls._ComfyInputParamDict
        setattr(newcls, 'INPUT_TYPES', INPUT_TYPES)
        
        if cls._HasIsChangedMethod:
            def IS_CHANGED(self, *args, **kwargs):
                proper_kws = proper_input(*args, **kwargs)
                return cls.IsChanged(**proper_kws)
            setattr(newcls, 'IS_CHANGED', IS_CHANGED)
        
        if cls._HasValidateInputMethod:
            @classmethod
            def VALIDATE_INPUTS(cls, *args, **kwargs):
                proper_kws = proper_input(*args, **kwargs)
                first_key = list(proper_kws.keys())[0]
                proper_kws.pop(first_key)  # remove the first key(self), cuz this is a classmethod
                return cls.ValidateInput(**proper_kws)
            setattr(newcls, 'VALIDATE_INPUTS', VALIDATE_INPUTS)

        def ON_DESTROY(ins:ComfyUINode):
            ins.__real_node_instance__.OnDestroy()
        setattr(newcls, 'ON_DESTROY', ON_DESTROY)
        
        cls._RealComfyUINodeCls = newcls
    
    def __init_subclass__(cls):
        if cls.IsAbstract is not None:
            cls._IsAbstract = cls.IsAbstract
        else:
            cls._IsAbstract = isabstract(cls)
        
        cls._InitBasicInfo()
        
        if not cls._IsAbstract:
            cls._CallSig = signature(cls.__call__)
            
            cls._InitFields()
            
            cls._ParamNameMap = {}
            for param_name, param in cls._InputFields.items():
                proper_name = param.proper_param_name or param_name
                cls._ParamNameMap[proper_name] = param_name
            
            cls._HasIsChangedMethod = cls.IsChanged != NodeBase.IsChanged
            if cls._HasIsChangedMethod:
                cls._HasIsChangedMethod = not is_empty_method(cls.IsChanged)
                
            cls._HasValidateInputMethod = cls.ValidateInput != NodeBase.ValidateInput
            if cls._HasValidateInputMethod:
                cls._HasValidateInputMethod = not is_empty_method(cls.ValidateInput)
            
            cls._InitTypesForComfy()
            
            cls._InitRealComfyUINode()
    
    # region internal use
    @classmethod
    def _MakeUIRetDict(cls, values: Tuple[Any]):
        '''return a dictionary for comfyUI to show the return values labeled with "UI"'''
        ret = {"ui":{}, "result": values}
        ui_dict = {}
        ui_ret_count = 0
        for i, (_, field) in enumerate(cls._ReturnFields):
            if 'ui' in field.tags:
                ui_val = values[i]
                if isinstance(ui_val, UI):
                    if not ui_val.ui_name in ui_dict:
                        ui_dict[ui_val.ui_name] = []
                    while len(ui_dict[ui_val.ui_name]) < ui_ret_count:
                        ui_dict[ui_val.ui_name].append(None)
                    ui_dict[ui_val.ui_name].append(ui_val._params)  # e.g. {'ui': {images: [{...}]}}    
                    
                    for extra_param_key, extra_param_val in ui_val._extra_params.items():
                        if not extra_param_key in ui_dict:
                            ui_dict[extra_param_key] = []
                        while len(ui_dict[extra_param_key]) < ui_ret_count:
                            ui_dict[extra_param_key].append(None)
                        ui_dict[extra_param_key].append(extra_param_val)
                else:
                    ComfyUILogger.warning(f'UI return value should be instance of `UI`, got {values[i]}')
                    
                ui_ret_count += 1
        
        ret['ui'].update(ui_dict)
        return ret
        
    @staticmethod
    def _AllSubclasses(non_abstract_only: bool = True)->Tuple[Type["NodeBase"]]:
        '''return all subclasses of `NodeBase`.'''
        all_clses = list(NodeBase.__subclasses__())
        subclses = set()
        while all_clses:
            cls = all_clses.pop()
            if not (non_abstract_only and cls._IsAbstract):
                subclses.add(cls)
            all_clses.extend(cls.__subclasses__())
        return tuple(subclses)
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
    def ValidateInput(cls, *args, **kwargs)->bool:   # type: ignore
        '''
        You can define this method to validate the node input is valid. Note that this is a classmethod
        (I think thats a bad idea actually, should be changed into instance method)
        
        return a boolean value to indicate whether the input is valid.
        '''

    def OnDestroy(self):
        '''This method will be called when the node is destroyed.'''



class StableRendererNodeBase(NodeBase):
    '''For inheritance category of stable-renderer nodes.'''
    
    Category = 'stable-renderer'
    
    NameSpace = "StableRenderer"



__all__.extend(['NodeBase', 'StableRendererNodeBase'])
