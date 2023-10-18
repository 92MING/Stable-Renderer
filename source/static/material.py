from static.enums import RenderOrder
from runtime.engineObj import EngineObj
from utils.base_clses import NamedObj
from static.texture import Texture
from static.shader import Shader
from typing import Union

class Material(EngineObj, NamedObj):

    _Default_Opaque_Material = None
    @classmethod
    def Default_Opaque_Material(cls):
        if cls._Default_Opaque_Material is None:
            cls._Default_Opaque_Material = Material('Default_Opaque_Material', Shader.Default_GBuffer_Shader())
        return cls._Default_Opaque_Material

    def __init__(self, name, shader:Shader=Shader, order:Union[RenderOrder, int]=RenderOrder.OPAQUE):
        NamedObj.__init__(self, name)
        self._shader = shader
        self._renderOrder = order.value if isinstance(order, RenderOrder) else order
        self._textures = {} # name: (texture, order)

    @property
    def shader(self)->Shader:
        return self._shader
    @property
    def drawAvailable(self) -> bool:
        return self._shader is not None
    @property
    def renderOrder(self)->int:
        return self._renderOrder
    @renderOrder.setter
    def renderOrder(self, value:Union[RenderOrder, int]):
        self._renderOrder = value.value if isinstance(value, RenderOrder) else value

    def addTexture(self, name:str, texture:Texture, order=None, replace=True):
        '''
        :param name: the name is the uniform name in shader
        :param texture: texture object
        :param order: the uniform order in shader. If None, the order will be the last one
        :param replace: if True, the texture will replace the existing one with the same order
        '''
        if order is None:
            order = len(self._textures)
        for name, texture_order in self._textures.items():
            _, order = texture_order
            if order == order:
                if not replace:
                    raise RuntimeError(f'Order {order} already exists in material {self.name}. Please remove it first.')
                else:
                    self._textures.pop(name)
                    break
        self._textures[name] = (texture, order)
    def removeTexture(self, name:str):
        self._textures.pop(name)
    def addDiffuseMap(self, texture:Texture):
        '''
        add diffuse map with default name=diffuseTex and order=0.
        If order 0 already exists, it will be replaced
        '''
        self.addTexture('diffuseTex', texture, order=0, replace=True)
    def addNormalMap(self, texture:Texture):
        '''
        add normal map with default name=normalTex and order=1.
        If order 1 already exists, it will be replaced
        '''
        self.addTexture('normalTex', texture, order=1, replace=True)

    def use(self):
        '''Use shader, and bind all textures to shader'''
        self._shader.useProgram()
        textures = [(name, tex, order) for name, (tex, order) in self._textures.items()]
        textures.sort(key=lambda x: x[2])
        for i, (name, tex, shader) in enumerate(textures):
            tex.bind(i, name)

__all__ = ['Material']