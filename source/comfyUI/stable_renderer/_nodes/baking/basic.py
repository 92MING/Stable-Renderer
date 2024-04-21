from ..node_base import StableRendererNodeBase

class BasicBaking(StableRendererNodeBase):
    Category = "baking"

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)