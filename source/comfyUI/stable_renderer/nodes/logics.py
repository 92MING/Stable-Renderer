from typing import Any

from .node_base import StableRendererNodeBase

class IsNotNoneNode(StableRendererNodeBase):
    
    Category = "Logic"
    
    def __call__(self, value: Any)->bool:
        return value is not None


class IfNode(StableRendererNodeBase):
    
    Category = "Logic"
    
    def __call__(self, condition: bool, true_value: Any, false_value: Any):
        pass