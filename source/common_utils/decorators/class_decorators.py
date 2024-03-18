'''
Class property makes the property workable for the class instead of the instance.

Note: `setter` is not supported for class property, due to the limitation of python.
'''

from collections.abc import Callable
from functools import update_wrapper
from typing import Type, Optional, TypeVar, Union, overload

from common_utils.global_utils import GetOrCreateGlobalValue

ClsT = TypeVar('ClsT', bound=type)
RetT = TypeVar('RetT', bound=type)

class class_property(property): # still inherit from property(but it works nothing for this class), cuz it makes `isinstance(..., property)` True
    '''
    Class property decorator. Acts like @property but for the class instead.
    Due to the limitation of python, @setter is not supported for class property.
    
    Example:
    ```python
    from typing import Type, Self
    class A:
        def __init__(self, k):
            self.k = k
        @class_property
        def a(cls):
            return cls(k=1)
    
    print(A.a.k)  # 1
    ''' 
    def __init__(self, fget:Callable[[ClsT], RetT]):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
    
    def __get__(self, _:None, owner: ClsT)->RetT:  # type: ignore
        return self.getter(owner)   # type: ignore

class abstract_class_property(class_property):
    '''
    Abstract class property decorator. Acts like @property but for the class instead.
    
    Example:
    ```python
    class A(ABC):
        @abstract_class_property
        def prop(cls):
            raise NotImplementedError()
    '''
    __isabstractmethod__ = True


class class_or_ins_property(property): # still inherit from property(but it works nothing for this class), cuz it makes `isinstance(..., property)` True
    '''
    Decorator for class or instance property. Acts like @property but also workable for the class instead.
    
    Example:
    ```python
    class A:
        @class_or_ins_method
        def func(cls_or_self):
            print(cls_or_self)
    
    A.func() # <class '__main__.A'>
    A().func() # <__main__.A object at ....>
    ```
    '''
    
    def __init__(self, fget:Callable[[ClsT], RetT]):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
    
    def __get__(self, instance:Optional[ClsT], owner:Type[ClsT])->RetT: # type: ignore
        if instance is None:
            return self.getter(owner)   # type: ignore
        else:
            return self.getter(instance)    # type: ignore

class class_or_ins_method(classmethod):
    def __get__(self, instance, owner):
        if not instance:
            return super().__get__(owner, owner)
        else:
            return super().__get__(owner, instance)



__all__ = ['class_property', 'abstract_class_property', 'class_or_ins_method', 'class_or_ins_property']


def prevent_re_init(cls: ClsT)->ClsT:
    '''
    Prevent re-init of a created instance.
    This decorator is useful when you want to make sure that the instance is only created once, e.g. singleton.
    '''
    
    origin_init = cls.__init__
    def new_init(self, *args, **kwargs):
        if hasattr(self, '__inited__') and getattr(self, '__inited__'):
            return  # do nothing
        setattr(self, '__inited__', True)
        origin_init(self, *args, **kwargs)       
    cls.__init__ = new_init
    return cls

@overload
def singleton(cls: ClsT)->ClsT:   # type: ignore
    '''
    The singleton class decorator.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    '''
@overload
def singleton(cross_module_singleton: bool=True)->Callable[[ClsT], ClsT]:   # type: ignore
    '''
    The singleton class decorator.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    
    if cross_module_singleton is True, the class will be unique even if you import the module in different ways.
    This is helpful when the import relationship is complex.
    '''

_cross_module_cls_dict = GetOrCreateGlobalValue('__CROSS_MODULE_CLASS_DICT__', dict)

def _singleton(cls: ClsT, cross_module_singleton:bool=False)->ClsT:
    '''
    Make the class as singleton class.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    '''
    if cross_module_singleton:
        cls_name = cls.__qualname__
        if cls_name in _cross_module_cls_dict:
            return _cross_module_cls_dict[cls_name].__instance__
    
    origin_new = cls.__new__
    def new_new(this_cls, *args, **kwargs):
        if hasattr(this_cls, '__instance__') and getattr(this_cls, '__instance__') is not None:
            return getattr(this_cls, '__instance__')
        if origin_new is object.__new__:
            return origin_new(this_cls) # type: ignore
        return origin_new(this_cls, *args, **kwargs)
    cls.__new__ = new_new
    
    origin_init = cls.__init__
    def new_init(self, *args, **kwargs):
        if hasattr(self.__class__, '__instance__') and getattr(self.__class__, '__instance__') is not None:
            return  # do nothing
        setattr(self.__class__, '__instance__', self)
        origin_init(self, *args, **kwargs)
    cls.__init__ = new_init
    
    if cross_module_singleton:
        _cross_module_cls_dict[cls_name] = cls
    
    return cls


def singleton(*args, **kwargs)->Union[ClsT, Callable[[ClsT], ClsT]]:  # type: ignore
    if len(args) + len(kwargs) != 1:
        raise ValueError('singleton only accept one argument.')
    
    arg = args[0] if args else None
    if arg is None:
        for _, v in kwargs.items():
            arg = v
            break
    
    if isinstance(arg, bool):
        return lambda cls: _singleton(cls, arg)
    else:
        return _singleton(arg)


__all__.extend(['prevent_re_init', 'singleton'])