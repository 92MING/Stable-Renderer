'''Base class of Game Material.'''
from static.enums import RenderOrder
from static.resourcesObj import ResourcesObj
from static.texture import Texture
from static.shader import Shader, Supported_Shader_Value_Types
from typing import Union, get_args, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from utils.type_utils import valueTypeCheck
from utils.global_utils import GetOrAddGlobalValue, SetGlobalValue
import glm

@dataclass
class DefaultTexture:
    name: str               # e.g. diffuseTex
    shader_check_name: str  # e.g. hasDiffuseTex
    slot:int                # e.g. 0
class DefaultTextureType(Enum):
    DiffuseTex = DefaultTexture('diffuseTex', 'hasDiffuseTex', 0)
    NormalTex = DefaultTexture('normalTex', 'hasNormalTex', 1)
    SpecularTex = DefaultTexture('specularTex', 'hasSpecularTex', 2)
    EmissionTex = DefaultTexture('emissionTex', 'hasEmissionTex', 3)
    OcclusionTex = DefaultTexture('occlusionTex', 'hasOcclusionTex', 4) # AO
    MetallicTex = DefaultTexture('metallicTex', 'hasMetallicTex', 5)
    RoughnessTex = DefaultTexture('roughnessTex', 'hasRoughnessTex', 6)
    DisplacementTex = DefaultTexture('displacementTex', 'hasDisplacementTex', 7)
    AlphaTex = DefaultTexture('alphaTex', 'hasAlphaTex', 8)
    @classmethod
    def FindDeafultTexture(cls, texName:str):
        '''Return DefaultTextureType with given name. Return None if not found.'''
        texName = texName.lower()
        for item in cls:
            if item.value.name.lower() == texName:
                return item
        return None
    @classmethod
    def FindDefaultTextureBySlot(cls, slot:int):
        '''Return DefaultTextureType with given slot. Return None if not found.'''
        for item in cls:
            if item.value.slot == slot:
                return item
        return None

@dataclass
class DefaultVariable:
    name: str
    val_type: type
class DefaultVariableType(Enum):
    Specular_Exp = DefaultVariable('Ns', float)
    Ambient_Col = DefaultVariable('Ka', glm.vec3)
    Diffuse_Col = DefaultVariable('Kd', glm.vec3)
    Specular_Col = DefaultVariable('Ks', glm.vec3)
    Emission_Col = DefaultVariable('Ke', glm.vec3)
    Optical_Density = DefaultVariable('Ni', float)
    Alpha = DefaultVariable('alpha', float)
    Illumination_Model = DefaultVariable('illum', int)


class Material(ResourcesObj):

    _BaseName = 'Material'

    # region default materials
    _Default_Opaque_Material = None
    _Default_Opaque_Material_Count = 0
    _Debug_Material = None
    _Debug_Material_Count = 0

    @classmethod
    def Default_Opaque_Material(cls, name=None)->'Material':
        '''Create a new material with default opaque shader.'''
        if name is None:
            name = f'Default_Opaque_Material_{cls._Default_Opaque_Material_Count}'
        elif name in cls.AllInstances():
            raise ValueError(f'Material name {name} already exists.')
        Material._Default_Opaque_Material_Count += 1
        return cls(name, Shader.Default_GBuffer_Shader())
    @classmethod
    def Debug_Material(cls, grayMode=False, pinkMode=False, whiteMode=False, name=None)->'Material':
        '''Create a new material with debug shader(which directly output color to screen space).'''
        if name is None:
            name = f'Debug_Material_{cls._Debug_Material_Count}'
        elif name in cls.AllInstances():
            raise ValueError(f'Material name {name} already exists.')
        Material._Debug_Material_Count += 1
        mat = cls(name, Shader.Debug_Shader())
        mat.setVariable('grayMode', grayMode)
        mat.setVariable('pinkMode', pinkMode)
        mat.setVariable('whiteMode', whiteMode)
        return mat
    # endregion

    # region cls method
    @classmethod
    def Load(cls, path, name=None, shader=None) -> Union['Material', Tuple['Material', ...]]:
        '''Try to load a material from given path.'''
        path, name = cls._GetPathAndName(path, name)
        format = path.split('.')[-1].lower()
        if format == "":
            raise ValueError(f'No format specified for {path}')
        subcls = cls.FindFormat(format)
        if subcls is None:
            raise ValueError(f'Not supported format {format} for {path}')
        subcls = subcls(name, shader)
        subcls.load(path)
        return subcls
    # endregion

    def __init__(self, name, shader:Shader=None, order:Union[RenderOrder, int]=RenderOrder.OPAQUE):
        '''
        :param name: name of the material
        :param shader: shader of the material. If None, will use Shader.Default_Defer_Shader()
        :param order: render order of the material.
        '''
        super().__init__(self, name)
        if shader is None:
            shader = Shader.Default_GBuffer_Shader() if not self.engine.IsDebugMode else Shader.Debug_Shader()
        self._shader:'Shader' = shader
        self._renderOrder = order.value if isinstance(order, RenderOrder) else order
        self._textures:Dict[str, Tuple['Texture', int]] = {} # name: (texture, slot)
        self._variables = {} # name: value
        self._id = GetOrAddGlobalValue("_MaterialCount", 0)
        SetGlobalValue("_MaterialCount", self._id+1)


    @property
    def id(self)->int:
        return self._id
    @property
    def shader(self)->Shader:
        return self._shader
    @shader.setter
    def shader(self, value:Shader):
        self._shader = value
    @property
    def drawAvailable(self) -> bool:
        '''Return True if the material is ready to draw(e.g. shader is not None).'''
        return self._shader is not None
    @property
    def renderOrder(self)->int:
        return self._renderOrder
    @renderOrder.setter
    def renderOrder(self, value:Union[RenderOrder, int]):
        self._renderOrder = value.value if isinstance(value, RenderOrder) else value

    def sendToGPU(self):
        for uniName, (tex, slot) in self._textures.items():
            tex.sendToGPU() # actually no need, since currently all Texture are sent to GPU on prepare() of ResourcesManager
    def clear(self):
        for uniName, (tex, slot) in self._textures.items():
            tex.clear()

    # region texture
    def addTexture(self, name:str, texture:Texture, order=None, replace=True):
        '''
        :param name: the name is the uniform name in shader
        :param texture: texture object
        :param order: the uniform order in shader. If None, the order will be the last one
        :param replace: if True, the texture will replace the existing one with the same order or name
        '''
        if name in self._textures and not replace:
            raise RuntimeError(f'Texture {name} already exists in material {self.name}. Please remove it first.')
        if order is None:
            order = len(self._textures)
        for other_name, (other_texture, other_order) in self._textures.items():
            if order == other_order:
                if not replace:
                    raise RuntimeError(f'Order {order} already exists in material {self.name}. Please remove it first.')
                else:
                    self._textures.pop(other_name)
                    break
        self._textures[name] = (texture, order)
    def removeTexture(self, name:str):
        self._textures.pop(name)
    def addDefaultTexture(self, texture:Texture, default_type:DefaultTextureType):
        '''
        add default texture with default name and order. Will replace the existing one with the same order & name.
        '''
        self.addTexture(default_type.value.name, texture, order=default_type.value.slot, replace=True)
    # endregion

    # region variables
    def setVariable(self, name:str, value:Supported_Shader_Value_Types):
        if not valueTypeCheck(value, Supported_Shader_Value_Types):
            raise TypeError(f'Unsupported value type {type(value)} for material variable {name}. Supported types are {get_args(Supported_Shader_Value_Types)}')
        self._variables[name] = value
    # endregion

    def use(self):
        '''Use shader, and bind all textures to shader'''
        self._shader.useProgram()
        textures = [(name, tex, order) for name, (tex, order) in self._textures.items()]
        textures.sort(key=lambda x: x[2])
        for i, (name, tex, _) in enumerate(textures):
            textureType = DefaultTextureType.FindDeafultTexture(name)
            if textureType is not None:
                # means this is a default texture
                self._shader.setUniform(textureType.value.shader_check_name, 1)
            else:
                if i < len(DefaultTextureType):
                    # means this is not a default texture, but it is in the default texture slot, so we need to set the default texture to None
                    defaultTexType = DefaultTextureType.FindDefaultTextureBySlot(i)
                    self._shader.setUniform(defaultTexType.value.shader_check_name, 0)
            tex.bind(i, self._shader.getUniformID(name))
        for name, value in self._variables.items():
            self._shader.setUniform(name, value)
        self._shader.setUniform('materialID', self.id) # for corresponding map


__all__ = ['Material', 'DefaultTextureType']