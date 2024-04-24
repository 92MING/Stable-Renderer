# -*- coding: utf-8 -*-
'''Decorators for caching functions/properties.'''

from weakref import WeakKeyDictionary, WeakValueDictionary, ref
from functools import lru_cache, wraps
from typing import Literal, overload, Callable, Union
from .class_decorators import class_property

class _IdKey:
    def __init__(self, value):
        self._id = id(value)
    def __hash__(self):
        return self._id
    def __eq__(self, other):
        return self._id == other._id
    def __repr__(self):
        return f"<IdKey(_id={self._id})>"

class _CachedFunc:
    
    def __init__(self, func, maxsize=128, typed=False):
        if not (callable(func) or isinstance(func, property) or isinstance(func, (classmethod, staticmethod))):
            raise TypeError(f'Expected a function or property, but got {func}({type})')
        self.lru_cache_args = (maxsize, typed)
        self.origin_func = func
        self.instances = WeakValueDictionary()
        self.methods = WeakKeyDictionary()
        self.func_type:Literal['Normal', 'Class', 'Static', 'Ins'] = 'Normal'
        self.is_property_type = False
        self.inited = False
        
    def __set_name__(self, owner, name):
        if isinstance(self.origin_func, classmethod):
            self.func_type = 'Class'
            self.origin_func = self.origin_func.__func__
        elif isinstance(self.origin_func, class_property):
            self.func_type = 'Class'
            self.origin_func = self.origin_func.fget
            self.is_property_type = True
        elif isinstance(self.origin_func, staticmethod):
            self.func_type = 'Static'
            self.origin_func = self.origin_func.__func__
        else:
            self.func_type = 'Ins'
            if isinstance(self.origin_func, property):
                self.origin_func = self.origin_func.fget
                self.is_property_type = True
        setattr(owner, name, self)
        
    def __get__(self, instance, owner):
        self.current_owner = owner
        self.current_instance = instance
        if not self.is_property_type:
            return self
        else:
            binding = self.current_instance if self.func_type == 'Ins' else self.current_owner
            key = _IdKey(binding)
            weakly_bound_cached_method = self.methods.get(key, None)    # type: ignore
            if weakly_bound_cached_method is None:
                self.instances[key] = binding
                _binder = ref(binding)
                
                @wraps(self.origin_func)    # type: ignore
                @lru_cache(*self.lru_cache_args)
                def weakly_bound_cached_method():
                    return self.origin_func(_binder())    # type: ignore
                
                self.methods[key] = weakly_bound_cached_method
                
            return weakly_bound_cached_method()
        
    def __call__(self, *args, **kwargs):
        if self.func_type in ('Class', 'Ins'):  # need to bind some stuff at the first argument
            binding = self.current_instance if self.func_type == 'Ins' else self.current_owner
            key = _IdKey(binding)
            weakly_bound_cached_method = self.methods.get(key, None)    # type: ignore
            if weakly_bound_cached_method is None:
                self.instances[key] = binding
                _binder = ref(binding)
                
                @wraps(self.origin_func)    # type: ignore
                @lru_cache(*self.lru_cache_args)
                def weakly_bound_cached_method(*args, **kwargs):
                    return self.origin_func(_binder(), *args, **kwargs)    # type: ignore
                
                self.methods[key] = weakly_bound_cached_method

            return weakly_bound_cached_method(*args, **kwargs)
        else:
            if not self.inited:
                self.inited = True
                self.func = lru_cache(*self.lru_cache_args)(self.origin_func)    # type: ignore
            return self.func(*args, **kwargs)

@overload
def cache(func:Callable): ...
@overload
def cache(prop:Union[property, class_property]): ...
@overload
def cache(maxsize:int=128, typed:bool=False): ...

def cache(*args, **kwargs):
    '''
    Cache any type of functions(normal/staticmethod/classmethod/instance method).
    @param maxsize: the max pool size of the cache(LRU)
    @param typed: whether to distinguish the type of the arguments, e.g. 1 and 1.0 are different when typed==True
    
    e.g.:
    ```
    @cache(maxsize=128, typed=False)    # can add arguments to the decorator
    def test(x):                        # normal function can be cached
        time.sleep(1)
        print('test', x)
        return x

    class A:
        @cache    # instance method can also be cached
        def f(self, x):
            print('f', x)
            time.sleep(1)
            return x

        @cache      # classmethod can also be cached
        @classmethod
        def g(cls, x):
            print('g', x)
            time.sleep(1)
            return x
        
        @cache    # staticmethod can also be cached
        @staticmethod
        def h(x):
            print('h', x)
            time.sleep(1)
            return x
            
        @cache      # property can also be cached
        @property
        def i(self):
            print('i')
            time.sleep(1)
            return 1
            
        @cache      # class_property can also be cached
        @class_property
        def j(cls):
            print('j')
            time.sleep(1)
            return 1
    ```
    '''
    if len(args) == 1 and (callable(args[0]) 
                           or isinstance(args[0], (classmethod, staticmethod))
                           or isinstance(args[0], property)):
        return _CachedFunc(args[0])
    else:
        real_args = {'maxsize': 128, 'typed': False}
        if args:
            real_args['maxsize'] = args[0]
            if len(args) > 1:
                real_args['typed'] = args[1]    # type: ignore
        real_args.update(kwargs)
        return lambda func: _CachedFunc(func, **real_args)


class cache_property(property):
    '''
    Cache a property value and set the cached value to the object by adding prefix and sufix to the property name.
    Default prefix and sufix are '__'(double underscore).
    '''

    @overload
    def __init__(func: Callable):...
    @overload
    def __init__(prefix:str="__", sufix:str="__"):...

def _cache_property(*args, **kwargs):
    '''
    Cache a property value and set the cached value to the object by adding prefix and sufix to the property name.
    Default prefix and sufix are '__'(double underscore).
    '''
    if (len(args) + len(kwargs)) == 1:
        func = args[0] if args else kwargs[tuple(kwargs.keys())[0]]
        return _cache_property("__", "__")(func)
    else:
        prefix = args[0] if len(args)>=1 else kwargs.get('prefix', "__")
        sufix = args[1] if len(args)>=2 else kwargs.get('sufix', "__")
        if prefix == sufix == "":
            raise ValueError("prefix and sufix can't be both empty.")
        def _decorator(func):
            @property
            @wraps(func)
            def _cached_property(self):
                key = prefix + func.__name__ + sufix
                if not hasattr(self, key):
                    setattr(self, key, func(self))
                return getattr(self, key)
            return _cached_property
        return _decorator

globals()['cache_property'] = _cache_property
    
__all__ = ['cache', 'cache_property']


