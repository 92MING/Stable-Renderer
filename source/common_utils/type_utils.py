# -*- coding: utf-8 -*-
'''common type related stuff'''

if __name__ == '__main__':  # for debugging
    import sys, os
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils'

from dataclasses import dataclass
from inspect import Parameter, signature, _empty, getmro
from collections import OrderedDict
from typing import (Any, Sequence, Union, ForwardRef, get_args as tp_get_args, get_origin as tp_get_origin, Callable, Awaitable, 
                    Iterable, Mapping, Literal, TypeAlias, Tuple, _SpecialForm, Type)
from types import UnionType
from inspect import getmro, signature
from typeguard import check_type as tg_check_type, TypeCheckError
from functools import cache
from pathlib import Path

from .path_utils import SOURCE_DIR

# region type checking
@cache
def _direct_check_sub_cls(sub_cls:Union[TypeAlias, str], super_cls:Union[TypeAlias, str]):
    if super_cls in (Any, any):
        return True
    
    sub_cls_origin = get_origin(sub_cls)
    super_cls_origin = get_origin(super_cls)
    
    if sub_cls_origin == Union and not super_cls_origin == Union:
        return all([_direct_check_sub_cls(arg, super_cls) for arg in get_args(sub_cls)])
    
    elif super_cls_origin == Union and not sub_cls_origin == Union:
        return any([_direct_check_sub_cls(sub_cls, arg) for arg in get_args(super_cls)])
    
    elif isinstance(sub_cls, ForwardRef):
        return _direct_check_sub_cls(sub_cls.__forward_arg__, super_cls)
    elif isinstance(super_cls, ForwardRef):
        return _direct_check_sub_cls(sub_cls, super_cls.__forward_arg__)
    
    if isinstance(sub_cls, str) and isinstance(super_cls, type):
        return sub_cls in [get_cls_name(cls) for cls in super_cls.__subclasses__()]
    if isinstance(sub_cls, type) and isinstance(super_cls, str):
        ret = super_cls in [get_cls_name(cls) for cls in getmro(sub_cls)]
        return ret
    
    elif isinstance(sub_cls, type) and isinstance(super_cls, type):
        return issubclass(sub_cls, super_cls)
    
    elif isinstance(sub_cls, str) and isinstance(super_cls, str):
        raise TypeError(f'Sub cls and super cls cannot both be str: sub_cls: {sub_cls}, super_cls: {super_cls}. There should be at least one type.')
    
    else:
        try:
            return issubclass(sub_cls, super_cls)   # type: ignore
        except Exception as e:
            raise e

def valueTypeCheck(value:Any, types: Union[str, TypeAlias, Sequence[Union[TypeAlias, str]]]):
    '''
    Check value with given types:
    E.g.:
    ```
    valueTypeCheck([1,2,'abc',1.23], list[int|str|float])
    valueTypeCheck([1,2], Any)
    valueTypeCheck(1, Literal[1, 2])
    valueTypeCheck(1, Union[int, str])
    valueTypeCheck(1, int | str)
    
    class A:
        pass
    a = A()
    valueTypeCheck(a, 'A') # True, accept class name
    ```
    '''
    if not isinstance(types, str) and isinstance(types, Sequence):
        return any(valueTypeCheck(value, t) for t in types)
    
    elif isinstance(types, str):
        return _direct_check_sub_cls(type(value), types)
    
    else:
        origin = get_origin(types)
        if origin not in (None, Union, Iterable, Literal):   
            # None: means no origin, e.g. list 
            # Union/UnionType: means union, e.g.: Union[int, str], int | str
            # Iterable: checking inner type of Iterable is meaningless, since it will destroy the structure of Iterable
            if issubclass(origin, Sequence):
                if not issubclass(origin, tuple):
                    args = get_args(types)
                    if len(args) == 0:  # no args, e.g. valueTypeCheck([1,2], list)
                        return isinstance(value, origin)
                    else:
                        return isinstance(value, origin) and all(valueTypeCheck(v, args[0]) for v in value)
                else:
                    args = get_args(types)
                    if len(args) == 0:
                        return isinstance(value, origin)
                    elif len(args) == 2 and args[-1] == Ellipsis:
                        return isinstance(value, origin) and all(valueTypeCheck(v, args[0]) for v in value)
                    else:
                        return (isinstance(value, origin)
                            and len(value) == len(get_args(types))
                            and all(valueTypeCheck(v, t) for v, t in zip(value, get_args(types))))  # type: ignore
            elif issubclass(origin, Mapping):
                return (isinstance(value, origin) 
                        and all(valueTypeCheck(v, get_args(types)[1]) for v in value.values())
                        and all(valueTypeCheck(k, get_args(types)[0]) for k in value.keys()))
            else:
                try:
                    tg_check_type(value, types)
                    return True
                except (TypeError, TypeCheckError):
                    return False
        else:
            try:
                tg_check_type(value, types)
                return True
            except (TypeError, TypeCheckError):
                return False

@cache  
def subClassCheck(sub_cls:Union[TypeAlias, Any], super_cls: Union[str, TypeAlias, Any, Sequence[Union[TypeAlias, str]]])->bool:
    '''
    Check if sub_cls is a subclass of super_cls.
    You could use `|` to represent Union, e.g.: `subClassCheck(sub_cls, int | str)`
    You could also use list to represent Union, e.g.: `subClassCheck(sub_cls, [int, str])`
    Class name is also supported, e.g.: `subClassCheck(sub_cls, 'A')`
    '''
    if not isinstance(super_cls, str) and isinstance(super_cls, Sequence):
        return any(_direct_check_sub_cls(sub_cls, t) for t in super_cls)
    try:
        return _direct_check_sub_cls(sub_cls, super_cls)
    except TypeError:
        return False

__all__ = ['subClassCheck', 'valueTypeCheck']

# endregion


# region helper functions
def get_origin(t:Any)->Union[Type, _SpecialForm, None]:
    '''
    Return the origin type of the type hint.
    Different to typing.get_origin, this function will convert some special types to their real origin type, 
    
    e.g. 
        * int|str -> Union                  (the origin typing.get_origin will return UnionType, which is not easy to do comparison)
        * ForwardRef('A') -> ForwardRef     (the origin typing.get_origin will return None, which is not correct)
        * _empty -> Any
    '''
    if t == _empty:
        return Any
    if isinstance(t, ForwardRef):
        return ForwardRef

    origin = tp_get_origin(t)
    if origin in (UnionType, Union):
        return Union    # type: ignore
    
    return origin

def get_args(t)->Tuple[Any, ...]:
    '''
    Return the args of the type hint.
    Different to typing.get_args, this function will convert some special types to their real args,
    
    e.g.
        * ForwardRef('A') -> ('A',)     (the origin typing.get_args will return (), which is not correct)
    '''
    if isinstance(t, ForwardRef):
        return (t.__forward_arg__,)
    return tp_get_args(t)

def get_cls_name(cls_or_ins: Any):
    '''
    Return the pure class name, without module name. e.g. 'A' instead of 'utils.xxx....A
    Preventing error when `__qualname__` is not available.
    '''
    if not type(cls_or_ins) == type:
        cls = type(cls_or_ins)
    else:
        cls = cls_or_ins
    if hasattr(cls, '__qualname__'):
        return cls.__qualname__
    else:
        return cls.__name__.split('.')[-1]

MAX_MRO_DISTANCE = 999

def get_mro_distance(cls:Any, super_cls:Union[type, str, None])->int:
    '''
    Return the distance of cls to super_cls in the mro.
    If cls is not a subclass of super_cls, return 999.
    
    Args:
        cls: the class to be checked.
        super_cls: the super class to be checked. It could also be special types like Union, Optional, ForwardRef, etc. 
    '''
    if cls is None and super_cls is None:
        return 0
    elif cls is None or super_cls is None:
        return MAX_MRO_DISTANCE
    
    if cls == Any:
        cls = object
    if super_cls == Any:
        super_cls = object
    if not subClassCheck(cls, super_cls):
        return MAX_MRO_DISTANCE
    
    origin = get_origin(super_cls)
    type_args = get_args(super_cls)
    
    if origin == Union and type_args:
        return min(get_mro_distance(cls, t) for t in type_args)
    
    elif origin == Literal and type_args:
        try:
            return (type_args == get_args(cls) and get_origin(cls)==Literal)   # e.g. Literal[1, 2, 3] == Literal[1, 2, 3] -> True
        except:
            return MAX_MRO_DISTANCE
    
    elif ((origin == ForwardRef and type_args) or isinstance(super_cls, str)):
        cls_mro_names = [get_cls_name(c) for c in getmro(cls)]
        try:
            return cls_mro_names.index(super_cls if isinstance(super_cls, str) else type_args[0])
        except ValueError:  # not found
            return MAX_MRO_DISTANCE
        
    else:
        try:
            return getmro(cls).index(super_cls)
        except ValueError:  # not found
            return MAX_MRO_DISTANCE


def get_proper_module_name(t: Any):
    '''
    Get the proper module name of the type.
    This is useful when running scripts directly for debugging, 
    
    e.g. you define a class in `utils.xxx....`, but the module will shows '__main__' when running the script directly.
    Class will be redefined by python as '__main__.A', which is different from 'utils.xxx....A'.
    By using this function, you could get the proper module name `utils.xxx....` instead of `__main__`
    '''
    module = t.__module__
    if module == "__main__":
        import __main__ as _main
        main_path = Path(_main.__file__).resolve()
        try:
            module = main_path.relative_to(SOURCE_DIR).with_suffix('').as_posix().replace('/', '.')
        except ValueError:
            module = '__main__' # not in source dir, use __main__ instead
    return module

__all__.extend(['get_origin', 'get_args', 'get_cls_name', 'get_mro_distance', 'get_proper_module_name'])

# endregion


# region functions type related
class InvalidParamError(TypeError):
    '''Invalid parameter error'''
    pass

@dataclass
class PackParamOutput:
    '''
    The output of function `pack_param`.
    The return value includes some useful information, for further usage.
    '''
    
    packed_params: Union[OrderedDict[str, Any], None]
    '''The packed params, e.g. {'a': 1, 'b': (2, 3), 'c': {'c1': 1, 'c2': 2}}'''
    func_params: OrderedDict[str, Parameter]
    '''The parameters of the function, e.g. {'a': <Parameter "a:int">, ...}, see inspect.Parameter for details'''
    var_args_field_name: Union[str, None]
    '''the `*args` field name. e.g. def f(*a) -> var_args_field_name="a"'''
    var_kwargs_field_name: Union[str, None]
    '''the `**kwargs` field name. e.g. def f(**a) -> var_kwargs_field_name="a"'''
    
def pack_param(origin_func: Callable, args, kwargs)-> PackParamOutput:
    '''
    Pack the args & kwargs to the format of the origin function.
    The return value includes some useful information. See `PackParamOutput` for details.
    
    e.g.
    ```python
    def f(a, *b, **c):
        pass
    print(pack_param(f, ('a', 'b1', 'b2'), {c1=1, c2=2}).packed_params) # {'a': 'a', 'b': ('b1', 'b2'), 'c': {'c1': 1, 'c2': 2}}
    ```
    '''
    sig = signature(origin_func)
    func_params = sig.parameters
    
    var_args_field_name: Union[str, None] = None
    var_kwargs_field_name: Union[str, None] = None
    
    for param_name, param in func_params.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            var_args_field_name = param_name
        elif param.kind == Parameter.VAR_KEYWORD:
            var_kwargs_field_name = param_name
    
    packed_params = OrderedDict()
    if var_args_field_name:
        packed_params[var_args_field_name] = []
    if var_kwargs_field_name:
        packed_params[var_kwargs_field_name] = {}
    
    for k, v in kwargs.items():
        if k in func_params:
            packed_params[k] = v
        else:
            if not var_kwargs_field_name:
                raise InvalidParamError(f'Unexpected kwargs: {k}={v}')
            else:
                packed_params[var_kwargs_field_name][k] = v
    
    args_copy = list(args)
    for name, param in func_params.items():
        if name in packed_params:
            continue
        
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
                if param.default == _empty:
                    raise InvalidParamError(f'Missing required parameter: {name}')
                packed_params[name] = param.default
    if len(args_copy) > 0:
        raise InvalidParamError(f'Too many positional arguments: {args_copy}')
    
    if var_args_field_name:
        packed_params[var_args_field_name] = tuple(packed_params[var_args_field_name])
        
    return PackParamOutput(packed_params=packed_params, 
                            func_params=OrderedDict(func_params), 
                            var_args_field_name=var_args_field_name,
                            var_kwargs_field_name=var_kwargs_field_name)

def func_param_type_check(func:Callable, *args, **kwargs)->bool:
    '''
    Check if the args and kwargs of func are valid.
    E.g.:
    ```
    def func(a:int, b:str, c):
        pass
    func_param_type_check(func, 1, 'abc', c=1.0) # True, for "c", since no type is specified, it will be Any
    ```
    '''
    # pack all args and kwargs into a dict
    try:
        pack_param(func, args, kwargs)
    except InvalidParamError:
        return False
    
    return True

__all__.extend(['pack_param', 'func_param_type_check'])
# endregion


# region common used types
BasicType = Union[int, float, str, bool, bytes, list, tuple, dict, set, type(None)]
'''Basic type of python, except for complex, range, slice, ellipsis, and types defined in typing module'''

AsyncFunc = Callable[..., Awaitable[Any]]
'''Async function'''

__all__.extend(['BasicType', 'AsyncFunc'])
# endregion
