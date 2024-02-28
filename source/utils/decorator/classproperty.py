'''
Class property makes the property workable for the class instead of the instance.

Note: `setter` is not supported for class property, due to the limitation of python.
'''

from collections.abc import Callable
from functools import update_wrapper
from typing import Type, overload, Optional, TypeVar, Union

ClsT = TypeVar('ClsT')
RetT = TypeVar('RetT')

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
        def a(cls:Type[Self]):
            return cls(k=1)
    
    print(A.a.k)  # 1
    ''' 
    def __init__(self, fget:Callable[[Type[ClsT]], RetT]):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
    
    def __get__(self, _:None, owner:Type[ClsT])->RetT:  # type: ignore
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
    
    def __init__(self, fget:Callable[[Union[Type[ClsT], ClsT]], RetT]):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
    
    @overload
    def __get__(self, instance:ClsT, owner:Type[ClsT])->RetT: ...   # type: ignore
    @overload
    def __get__(self, _:None, owner:Type[ClsT])->RetT: ...  # type: ignore
    
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
