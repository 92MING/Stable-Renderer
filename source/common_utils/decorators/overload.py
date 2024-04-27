'''`Overload` make you able to define multiple functions with the same name but different parameters.'''

if __name__ == "__main__":
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils.decorators'

from inspect import signature, Parameter, getmro, _empty
from enum import Enum
from collections import OrderedDict
from typing import Callable, get_args, overload as tp_overload, Any, Literal, List, Tuple, Optional, Union, TypeAlias
from types import MethodType

from ..type_utils import get_origin, get_mro_distance, get_proper_module_name, valueTypeCheck
from ..global_utils import GetOrCreateGlobalValue


_overload_funcs_dict:dict[tuple[str, Optional[str], str], list['_Overload']] = GetOrCreateGlobalValue("__OVERLOAD_FUNCS", dict)
'''{(module, class, func name): [overload 1, ...]}}'''


class _WrongInputError(Exception):
    pass
class NoSuchFunctionError(Exception):
    pass
class WrongCallingError(Exception):
    pass

Overload = tp_overload # make IDE think Overload is a typing.overload
'''
Class method overload. Must be used inside class definition.

Example:
```
    class A:
        @Overload
        def f(self, x:int):
            print('int x:', x)
        @Overload
        def f(self, x:str):
            print('str x:', x)
    a=A()
    a.f(1)
    a.f('1')
```
'''

class _FuncType(Enum):
    NORMAL_FUNC = 0
    '''Normal function that doesn't belong to any class.'''
    CLASS_FUNC = 1
    '''Class function that belongs to a class.'''
    METHOD = 2
    '''Method from an class instance'''
    STATICMETHOD = 3
    '''`staticmethod` that belongs to a class.'''
    CLASSMETHOD = 4
    '''`classmethod` that belongs to a class.'''

MAX_MRO_DISTANCE = 999

class _Overload:
    
    getter: Union[object, None]
    '''The instance or class that calls the function.'''
    
    func_name: str
    '''The name of the function.'''
    origin_class: Union[str, None] = None
    '''The class that the function belongs to. If `None`, the function doesn't belong to any class.'''
    origin_module: str
    '''The module that the function belongs to.'''
    func_type: _FuncType
    '''The type of the function. See `_FuncType` for details.'''
    
    origin_func: Callable
    '''The original function.'''
    
    parameters: OrderedDict[str, Parameter]
    '''The parameters of the function.'''
    annotations: OrderedDict[str, Union[type, Any]]
    '''The tidied up annotations of the function, e.g. changing _empty to Any, Self to real class, etc.'''
    var_args_field_name: Union[str, None] = None
    '''The name of the field that stores the var_args. If `None`, the function doesn't have *args.'''
    var_kwargs_field_name: Union[str, None] = None
    '''The name of the field that stores the var_kwargs. If `None`, the function doesn't have **kwargs.'''
    
    @property
    def func_key(self)->tuple[str, Optional[str], str]:
        return (self.origin_module, self.origin_class if self.origin_class else None, self.func_name)
    
    @property
    def _is_under_class(self):
        return self.origin_class is not None
    
    
    def __init__(self, func: Callable):
        self.getter = None
        self.origin_func = func
        
        func_names = func.__qualname__.split('.')
        self.func_name = func_names[-1]
        if len(func_names)> 1:
            self.origin_class = func_names[-2]
        self.origin_module = func.__module__
        if self.origin_module == '__main__':    # special treatment for debug running
            self.origin_module = get_proper_module_name(func)
        
        if isinstance(func, MethodType):
            self.func_type = _FuncType.METHOD
        elif isinstance(func, classmethod):
            self.func_type = _FuncType.CLASSMETHOD
        elif isinstance(func, staticmethod):
            self.func_type = _FuncType.STATICMETHOD
        elif self.origin_class:
            self.func_type = _FuncType.CLASS_FUNC
        else:
            self.func_type = _FuncType.NORMAL_FUNC     # will becomes CLASS_FUNC if founds in `__set_name__`
        
        if self.func_key not in _overload_funcs_dict:
            _overload_funcs_dict[self.func_key] = []
        
        self.parameters = signature(func).parameters.copy()   # type: ignore
        
        for param_name in self.parameters:
            if self.parameters[param_name].kind == Parameter.VAR_POSITIONAL:
                self.var_args_field_name = param_name
            elif self.parameters[param_name].kind == Parameter.VAR_KEYWORD:
                self.var_kwargs_field_name = param_name
        
        self._tidy_up_annotations()
        
        _overload_funcs_dict[self.func_key].append(self)

    def __get__(self, instance, owner):
        if self.func_type == _FuncType.CLASS_FUNC:
            self.getter = instance
        elif self.func_type == _FuncType.CLASSMETHOD:
            self.getter = owner
        else:
            self.getter = None # means no need to give instance
        return self
 
    def _tidy_up_single_anno(self, anno):
        if anno == _empty:
            return Any
        elif (origin_type:=get_origin(anno)) and (type_args:=get_args(anno)):
            tidied_up_args = [self._tidy_up_single_anno(arg) for arg in type_args]
            return origin_type[tuple(tidied_up_args)]
        return anno
        
    def _tidy_up_annotations(self):
        self.annotations = OrderedDict()
        for param_name, param in self.parameters.items():   # assume self.parameters is set
            anno = self._tidy_up_single_anno(param.annotation)
            self.annotations[param_name] = anno

    def _pack_param(self, args, kwargs)-> Union[OrderedDict[str, Any], None]:
        packed_params = OrderedDict()
        if self.var_args_field_name:
            packed_params[self.var_args_field_name] = []
        if self.var_kwargs_field_name:
            packed_params[self.var_kwargs_field_name] = {}
        
        for k, v in kwargs.items():
            if k in self.parameters:
                packed_params[k] = v
            elif not self.var_kwargs_field_name:
                return None # means the kwargs contains some unexpected params, not suitable for this function
            else:
                packed_params[self.var_kwargs_field_name][k] = v
        
        args_copy = list(args)
        for name, param in self.parameters.items():
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
                elif name in kwargs:
                    packed_params[name] = kwargs[name]
                else:
                    if param.default == _empty:
                        return None # means the args are not enough for this function, not suitable for this function
                    packed_params[name] = param.default
        if len(args_copy) > 0:
            return None # means the args are too many for this function, not suitable for this function
        
        if self.var_args_field_name:
            packed_params[self.var_args_field_name] = tuple(packed_params[self.var_args_field_name])
            
        return packed_params
    
    def _calculate_mro_distance(self, 
                                correct_type: Union[type, TypeAlias], 
                                val: Any, 
                                default: Any,
                                mode: Literal['args', 'kwargs', 'normal'])->int:
        if val == default:
            return 0
        if mode == 'normal':
            dis = get_mro_distance(type(val), correct_type) # type: ignore
        elif mode == 'args':
            dis = 0
            for v in val:
                dis += get_mro_distance(type(v), correct_type) # type: ignore
        else:
            dis = 0
            for _, v in val.items():
                dis += get_mro_distance(type(v), correct_type) # type: ignore
        return dis
    
    def _calculate_total_param_mro_distance(self, packed_params: Optional[OrderedDict[str, Any]])->int:
        if packed_params is None:
            return MAX_MRO_DISTANCE
        
        mro_dis = 0
        for param_name, val in packed_params.items():
            
            param_correct_type = self.annotations[param_name]
            default=self.parameters[param_name].default
            if param_name == self.var_args_field_name: # *args
                mro_dis += self._calculate_mro_distance(correct_type=param_correct_type, 
                                                        val=val,
                                                        default=default,
                                                        mode='args')
            elif param_name == self.var_kwargs_field_name: # **kwargs
                mro_dis += self._calculate_mro_distance(correct_type=param_correct_type, 
                                                        val=val, 
                                                        default=default,
                                                        mode='kwargs')
            else:   # normal param
                mro_dis += self._calculate_mro_distance(correct_type=param_correct_type, 
                                                        val=val, 
                                                        default=default,
                                                        mode='normal')
        return mro_dis
    
    def _internal_call(self, 
                       possible_funcs:List['_Overload'], 
                       args, 
                       kwargs)->Tuple[bool, Any]:
        suitable_funcs = []
        
        if self.func_type in (_FuncType.CLASS_FUNC, _FuncType.CLASSMETHOD):
            args = (self.getter, *args)
        
        for overload_func in possible_funcs:
            packed_params = overload_func._pack_param(args, kwargs) # packed_params can be None, means not suitable for this function
            if packed_params is not None:
                mro_dis = overload_func._calculate_total_param_mro_distance(packed_params) / len(overload_func.parameters)
                suitable_funcs.append((overload_func, mro_dis))
        
        if suitable_funcs:
            suitable_funcs = sorted(suitable_funcs, key=lambda x: x[1])
            overload_func: '_Overload' = suitable_funcs[0][0]
            return (True, overload_func.origin_func(*args, **kwargs))
        else:
            return (False, None)
    
    def __call__(self, *args, **kwargs):
        if self._is_under_class:
            
            possible_classes = getmro(self.getter.__class__)  # search overload funcs in all base classes
            possible_funcs = []
            
            for possible_cls in possible_classes:
                module_name = get_proper_module_name(possible_cls)
                key = (module_name, possible_cls.__qualname__, self.func_name)    
                
                if key in _overload_funcs_dict:
                    possible_funcs.extend(_overload_funcs_dict[key])
                    
        else:
            key = (self.origin_module, None, self.func_name)
            possible_funcs = _overload_funcs_dict.get(key, [])
        
        result, val = self._internal_call(possible_funcs, args, kwargs)
        if result:
            return val
        
        func_name = f'{self.origin_class}.{self.func_name}' if self.origin_class else self.func_name
        raise NoSuchFunctionError(f'No such function:{func_name} with args: {args} and kwargs:{kwargs}')



locals()['Overload'] = _Overload    
# set name 'Overload' back to the class _Overload
# This is for faking the IDE to think Overload is a typing.overload, so as to enable the overload type hinting.



__all__ = ['Overload', 'NoSuchFunctionError', 'WrongCallingError']



if __name__ == '__main__':  # for debug
    from typing import Optional
    class B:
        pass
    class A:
        @Overload
        def f(self, x:str):
            '''str'''
            print('str x:', x)
        
        @Overload
        def f(self, x: Optional["B"]=None): # type: ignore
            '''int'''
            print('B type x:', x)
        
    b = B()
    a = A()
    a.f(x=b)