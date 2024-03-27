from .node_base import NodeBase

from .maths import *
from .io import *
from .loaders import *
from .samplers import *
from .schedulers import *
from .processing import *

class TestNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "x": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1})
                    },
                }

    OUTPUT_NODE = True
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("x", "y")
    CATEGORY = "Stable-Renderer"
    FUNCTION = "__call__"
    
    def __call__(self, x: int):
        data = {"filename":"animate_diff_00004_.gif", "type": "output"}
        return {"ui": {"images": [data, data], "animated": (True, True)}, "result": (x, x+1)}
        

# for ComfyUI's node registry
NODE_CLASS_MAPPINGS = {cls.__qualname__: cls._RealComfyUINode for cls in NodeBase._AllSubclasses() if not cls._IsAbstract}
NODE_DISPLAY_NAME_MAPPINGS = {cls.__qualname__: cls._ReadableName for cls in NodeBase._AllSubclasses() if not cls._IsAbstract}

NODE_CLASS_MAPPINGS["TestNode"] = TestNode
NODE_DISPLAY_NAME_MAPPINGS["TestNode"] = "Test Node"