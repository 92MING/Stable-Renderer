from .node_base import NodeBase

from .samplers import *


# for ComfyUI's node registry
NODE_CLASS_MAPPINGS = {cls.__qualname__: cls._RealComfyUINode for cls in NodeBase._AllSubclasses() if not cls._IsAbstract}
NODE_DISPLAY_NAME_MAPPINGS = {cls.__qualname__: cls._ReadableName for cls in NodeBase._AllSubclasses() if not cls._IsAbstract}

