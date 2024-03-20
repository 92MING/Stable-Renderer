from typing import Any, Literal

from .node_base import StableRendererNodeBase
from comfyUI.types import *


class IsNotNoneNode(StableRendererNodeBase):
    '''
    Check whether the value is not None.
    
    Args:
        - value: The value to check.
        - mode: The mode to use. If 'strict', the node will only return True if the value is not None. If 'normal', the node will return True if the value is not None and not False.
    '''
    
    Category = "Logic"
    
    def __call__(self, value: Any, mode: Literal['strict', 'normal']='normal')->bool:
        if mode == 'strict':
            return value is not None    # strict mode only returns True if the value is not None
        return bool(value)


class IfNode(StableRendererNodeBase):
    '''
    If the condition is True, return the true value. Otherwise, return the false value.
    
    Args:
        - condition: The condition to check.
        - true_value: The value to return if the condition is True.
        - false_value: The value to return if the condition is False.
        
    Note: value will be evaluated lazily, not need to worry about the performance.
    '''
    
    Category = "Logic"
    
    def __call__(self, 
                 condition: bool, 
                 true_value: Lazy[Any], 
                 false_value: Lazy[Any])->Any: 
        if condition:
            return true_value.value
        return false_value.value
