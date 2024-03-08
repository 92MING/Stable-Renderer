from typing import Callable, Type, TypeVar
from functools import update_wrapper

ClsT = TypeVar('ClsT')
RetT = TypeVar('RetT')

class classproperty(property): # still inherit from property(but it works nothing for this class), cuz it makes `isinstance(..., property)` True
    def __init__(self, fget:Callable[[Type[ClsT]], RetT]):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
    
    def __get__(self, _:None, owner:Type[ClsT])->RetT:  # type: ignore
        return self.getter(owner)   # type: ignore
    
    
__all__ = ['classproperty']