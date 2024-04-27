from typing import Any, Literal

from comfyUI.types import *
from common_utils.type_utils import get_cls_name
from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import is_verbose_mode


class IsNotNoneNode(StableRendererNodeBase):
    '''
    Check whether the value is not None.
    
    Args:
        - value: The value to check.
        - mode(strict/normal): 
            If 'strict', the node will only return True when eval(value is not None). 
            If 'normal', the node will let the bool(value) to decide the result.
    '''
    
    Category = "Logic"
    
    def __call__(self, value: Any, mode: Literal['strict', 'normal'] = 'strict')->bool:
        if mode == 'strict':
            result = value is not None    # strict mode only returns True if the value is not None
        else:
            try:
                result = bool(value)
            except:
                result = value is not None

        if is_verbose_mode():
                val_str = str(value)
                if len(val_str) > 18:
                    val_str = val_str[:15] + '...'
                ComfyUILogger.debug(f'IsNotNoneNode: value={val_str}, mode={mode}, result={result}')
        return result

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
            if is_verbose_mode():
                val = true_value.value
                str_val = str(val)
                if len(str_val) > 18:
                    str_val = str_val[:15] + '...'
                ComfyUILogger.debug(f'IfNode: condition=True, return true_value={str_val}')
                return val
            else:
                return true_value.value
        else:
            if is_verbose_mode():
                val = false_value.value
                str_val = str(val)
                if len(str_val) > 18:
                    str_val = str_val[:15] + '...'
                ComfyUILogger.debug(f'IfNode: condition=False, return false_value={str_val}')
                return val
            else:
                return false_value.value

class IfValTypeEqual(StableRendererNodeBase):
    '''check if the income value's type equals to your given string'''
    
    Category = "Logic"
    
    def __call__(self, val: Any, type_name: str)->bool:
        val_cls_name = get_cls_name(val)
        return val_cls_name.upper() == type_name.upper()

    
__all__ = ['IsNotNoneNode', 'IfNode', 'IfValTypeEqual']