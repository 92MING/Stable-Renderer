from comfyUI.types import *


class TextConcat(StableRenderingNode):
    Category = "processing"

    def __call__(self,
                 text_a: STRING(forceInput=True),
                 text_b: STRING(),) -> STRING:
        """Concatenate text_a with text_b"""
        return text_a + text_b


class TextReplace(StableRenderingNode):
    Category = "processing"

    def __call__(self,
                 text: STRING(forceInput=True),
                 pattern: STRING(forceInput=True),
                 replace: STRING(forceInput=True),
    ) -> STRING:
        """Replace `pattern` in `string` with `replace`"""
        return text.replace(pattern, replace)


__all__ = ["TextConcat", 'TextReplace']