'''Base class of Game Material.'''
import glm

from typing import Union, get_args, Dict, Tuple, TypeVar, Type, Optional, Literal
from enum import Enum
from dataclasses import dataclass
from common_utils.type_utils import valueTypeCheck
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue
from attr import attrib, attrs

from engine.static.enums import RenderOrder
from engine.static.resources_obj import ResourcesObj
from engine.static.texture import Texture
from engine.static.shader import Shader, SupportedShaderValueTypes


T = TypeVar('T', bound='Material')
'''Type var for child class of Material'''

@dataclass
class DefaultTexture:
    '''This class is used to store the default texture info in glsl.'''
    name: str
    '''Name of the default texture, e.g. diffuseTex'''
    shader_check_name: str
    '''for glsl internal use, e.g. "hasDiffuseTex"'''
    slot:int
    '''glsl slot of this default texture, e.g. 0 for diffuseTex'''

class DefaultTextureType(Enum):
    '''This class is used to store the default texture types. e.g. DiffuseTex, NormalTex, etc.'''

    DiffuseTex = DefaultTexture('diffuseTex', 'hasDiffuseTex', 0)
    '''Color texture of the material'''
    NormalTex = DefaultTexture('normalTex', 'hasNormalTex', 1)
    '''Normal texture of the material'''
    SpecularTex = DefaultTexture('specularTex', 'hasSpecularTex', 2)
    '''Light reflection texture of the material'''
    EmissionTex = DefaultTexture('emissionTex', 'hasEmissionTex', 3)
    '''Light emission texture of the material'''
    OcclusionTex = DefaultTexture('occlusionTex', 'hasOcclusionTex', 4)
    '''Ambient occlusion texture of the material. Known as `AO` texture.'''
    MetallicTex = DefaultTexture('metallicTex', 'hasMetallicTex', 5)
    '''Metallic texture of the material'''
    RoughnessTex = DefaultTexture('roughnessTex', 'hasRoughnessTex', 6)
    '''Roughness texture of the material'''
    DisplacementTex = DefaultTexture('displacementTex', 'hasDisplacementTex', 7)
    '''Displacement texture of the material'''
    AlphaTex = DefaultTexture('alphaTex', 'hasAlphaTex', 8)
    '''Alpha texture of the material'''
    
    NoiseTex = DefaultTexture('noiseTex', 'hasNoiseTex', 9)
    '''Noise texture, for AI rendering'''

    @classmethod
    def FindDefaultTexture(cls, texName:str):
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
    '''This class is used to represent a default glsl variable'''
    name: str
    val_type: type

class DefaultVariableType(Enum):
    '''This class is used to store the default variable types in glsl. e.g. Ns, Ka, Kd, etc.'''

    Specular_Exp = DefaultVariable('Ns', float)
    '''Specular exponent of the material'''
    Ambient_Col = DefaultVariable('Ka', glm.vec3)
    '''Ambient color of the material'''
    Diffuse_Col = DefaultVariable('Kd', glm.vec3)
    '''Diffuse color of the material'''
    Specular_Col = DefaultVariable('Ks', glm.vec3)
    '''Specular color of the material'''
    Emission_Col = DefaultVariable('Ke', glm.vec3)
    '''Emission color of the material'''
    Optical_Density = DefaultVariable('Ni', float)
    '''Optical density of the material'''
    Alpha = DefaultVariable('alpha', float)
    '''Alpha value of the material'''
    Illumination_Model = DefaultVariable('illum', int)
    '''Illumination model of the material'''

def _new_mat_id():
    mat_id = GetOrAddGlobalValue('__ENGINE_MATERIAL_INT_ID__', 0)
    SetGlobalValue('__ENGINE_MATERIAL_INT_ID__', mat_id+1)  # type: ignore
    return mat_id

@attrs(eq=False, repr=False)
class Material(ResourcesObj):

    BaseClsName = 'Material'

    @classmethod
    def DefaultOpaqueMaterial(cls, **kwargs):
        '''Create a new material with default opaque shader.'''
        return cls(shader=Shader.Default_GBuffer_Shader(), **kwargs)

    @classmethod
    def DefaultDebugMaterial(cls, 
                             mode: Literal['pink', 'white', 'grayscale']='pink',
                             **kwargs):
        '''
        Create a new material with debug shader(which directly output color to screen space).
        
        mode:
            - pink: turn missing texture to pink
            - white: turn missing texture to white
            - grayscale: all pixels are grayscale
        '''
        mat = cls(shader=Shader.Debug_Shader(), **kwargs)
        mode_map = {
            'pink': 0,
            'white': 1,
            'grayscale': 2
        }
        mat.setVariable('mode', mode_map[mode.strip().lower()])
        return mat

    @classmethod
    def Load(cls: Type[T], path, *args, **kwargs) -> Union[T, Tuple[T]]:
        '''
        Try to load a material from given path.

        Each subclass should override this method to provide the corresponding load method.
        '''
        format = path.split('.')[-1].lower()
        if format == "":
            raise ValueError(f'No format specified for {path}')
        subcls = Material.FindFormatCls(format)
        if subcls is None:
            raise ValueError(f'No supported class for format `{format}`')
        return subcls.Load(path, *args, **kwargs)
    
    shader: Shader = attrib(default=None, kw_only=True)
    '''Shader of the material'''
    render_order: Union[RenderOrder, int] = attrib(default=RenderOrder.OPAQUE, kw_only=True)       
    '''Render order of the material'''
    variables: Dict[str, SupportedShaderValueTypes] = attrib(factory=dict, kw_only=True)
    '''Shader variables of the material'''
    textures: Dict[str, Tuple['Texture', int]] = attrib(factory=dict, kw_only=True)
    '''Textures of the material'''
    _mat_id: int = attrib(factory=_new_mat_id, init=False)
    '''special material id for corresponding map'''
    
    def __attrs_post_init__(self):
        if self.shader is None:
            self.shader = Shader.Default_GBuffer_Shader() if not self.engine.IsDebugMode else Shader.Debug_Shader()
    
    @property
    def mat_id(self):
        '''Return the material id. This is for corresponding map.'''
        return self._mat_id
    
    @property
    def draw_available(self) -> bool:
        '''Return True if the material is ready to draw(e.g. shader is not None).'''
        return self.shader is not None

    @property
    def loaded(self) -> bool:
        '''Return True if the material is loaded.'''
        return self.shader is not None and all([tex.loaded for tex, _ in self.textures.values()])
    
    def load(self):
        for uniName, (tex, slot) in self.textures.items():
            tex.load() # actually no need, since currently all Texture are sent to GPU on prepare() of ResourcesManager
            
    def clear(self):
        for uniName, (tex, slot) in self.textures.items():
            tex.clear()

    def addTexture(self, name:str, texture:Texture, order=None, replace=True):
        '''
        :param name: the name is the uniform name in shader
        :param texture: texture object
        :param order: the uniform order in shader. If None, the order will be the last one
        :param replace: if True, the texture will replace the existing one with the same order or name
        '''
        if name in self.textures and not replace:
            raise RuntimeError(f'Texture {name} already exists in material {self._name}. Please remove it first.')
        if order is None:
            order = len(self.textures)
        for other_name, (other_texture, other_order) in self.textures.items():
            if order == other_order:
                if not replace:
                    raise RuntimeError(f'Order {order} already exists in material {self._name}. Please remove it first.')
                else:
                    self.textures.pop(other_name)
                    break
        self.textures[name] = (texture, order)

    def removeTexture(self, name:str):
        self.textures.pop(name)

    def addDefaultTexture(self, texture:Texture, default_type:DefaultTextureType):
        '''
        add default texture with default name and order. Will replace the existing one with the same order & name.
        '''
        self.addTexture(default_type.value.name, texture, order=default_type.value.slot, replace=True)
    
    def setVariable(self, name:str, value:SupportedShaderValueTypes):
        if not valueTypeCheck(value, SupportedShaderValueTypes):    # type: ignore
            raise TypeError(f'Unsupported value type {type(value)} for material variable {name}. Supported types are {get_args(SupportedShaderValueTypes)}')
        self.variables[name] = value
    
    def use(self):
        '''Use shader, and bind all textures to shader'''
        self.shader.useProgram()
        textures = [(name, tex, order) for name, (tex, order) in self.textures.items()]
        textures.sort(key=lambda x: x[2])
        for i, (name, tex, _) in enumerate(textures):
            textureType = DefaultTextureType.FindDefaultTexture(name)
            if textureType is not None:
                # means this is a default texture
                self.shader.setUniform(textureType.value.shader_check_name, 1)
            else:
                if i < len(DefaultTextureType):
                    # means this is not a default texture, but it is in the default texture slot, so we need to set the default texture to None
                    defaultTexType = DefaultTextureType.FindDefaultTextureBySlot(i)
                    self._shader.setUniform(defaultTexType.value.shader_check_name, 0)  # type: ignore
            tex.bind(i, self.shader.getUniformID(name))
        for name, value in self.variables.items():
            self.shader.setUniform(name, value)
        self.shader.setUniform('materialID', self.mat_id)



__all__ = ['Material', 'DefaultTextureType']