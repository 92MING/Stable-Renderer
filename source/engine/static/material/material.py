'''Base class of Game Material.'''
import OpenGL.GL as gl
from typing import Union, get_args, Dict, Tuple, TypeVar, Type, Literal, overload, TYPE_CHECKING
from attr import attrib, attrs

from common_utils.type_utils import valueTypeCheck
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue, is_dev_mode 
from common_utils.debug_utils import EngineLogger
from engine.static.enums import RenderOrder, DefaultTextureType, DefaultVariableType, EngineFBO
from engine.static.resources_obj import ResourcesObj
from engine.static.texture import Texture
from engine.static.shader import Shader, SupportedShaderValueTypes

if TYPE_CHECKING:
    from ..corrmap import CorrespondMap


T = TypeVar('T', bound='Material')
'''Type var for child class of Material'''


def _new_mat_id():
    '''
    Get new material ID.
    Please note that id 0 is reserved, but currently no meaning.
    '''
    mat_id = GetOrAddGlobalValue('__ENGINE_MATERIAL_INT_ID__', 1)
    SetGlobalValue('__ENGINE_MATERIAL_INT_ID__', mat_id+1)  # type: ignore
    return mat_id

__TEXTURE_BINDING_OFFSET__ = len(EngineFBO.__members__)
def _maximum_tex_image_units():
    return gl.glGetIntegerv(gl.GL_MAX_TEXTURE_IMAGE_UNITS)

@attrs(eq=False, repr=False)
class Material(ResourcesObj):

    BaseClsName = 'Material'

    @classmethod
    def DefaultOpaqueMaterial(cls, **kwargs):
        '''
        Create a new material with default opaque shader.
        It will be shaded on GBuffer pass.
        '''
        return cls(shader=Shader.DefaultGBufferShader(), **kwargs)
    
    @classmethod
    def DefaultTransparentMaterial(cls, **kwargs):
        '''
        Create a new material with default to be shaded in transparent order queue.
        It will be shaded on GBuffer pass.
        '''
        return cls(shader=Shader.DefaultGBufferShader(), render_order=RenderOrder.TRANSPARENT, **kwargs)

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
        mat = cls(shader=Shader.DebugShader(), **kwargs)
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
        subcls: Type[T] = Material.FindFormatCls(format)   # type: ignore
        if subcls is None:
            raise ValueError(f'No supported class for format `{format}`')
        return subcls.Load(path, *args, **kwargs)
    
    shader: Shader = attrib(default=None, kw_only=True)
    '''Shader of the material'''
    render_order: Union[RenderOrder, int] = attrib(default=RenderOrder.OPAQUE, kw_only=True)       
    '''Render order of the material'''
    variables: Dict[str, SupportedShaderValueTypes] = attrib(factory=dict, kw_only=True)
    '''Shader variables of the material'''
    textures: Dict[str, Tuple[Union['Texture', "CorrespondMap"], int]] = attrib(factory=dict, kw_only=True)
    '''Textures of the material'''
    _mat_id: int = attrib(factory=_new_mat_id, init=False)
    '''special material id for corresponding map. Internal use only.'''
    
    def __attrs_post_init__(self):
        if self.shader is None:
            self.shader = Shader.DefaultGBufferShader() if not self.engine.IsDebugMode else Shader.DebugShader()
    
    @property
    def materialID(self):
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
        for _, (tex, _) in self.textures.items():
            tex.load() # actually no need, since currently all Texture are sent to GPU on prepare() of ResourcesManager
            
    def clear(self):
        for _, (tex, _) in self.textures.items():
            tex.clear()

    def addTexture(self, name:str, texture:Union[Texture, "CorrespondMap"], slot: int|None=None, replace=True):
        '''
        Args:
            - name: the name is the uniform name in shader
            - texture: texture object
            - order: the uniform order in shader. If not given, it will be the smallest empty slot in this material
            - replace: if True, the texture will replace the existing one with the same slot or name
        
        Note: for adding correspondence map, please use `addDefaultTexture` instead.
        '''
        if name in self.textures and not replace:
            raise RuntimeError(f'Texture {name} already exists in material {self._name}. Please remove it first.')
        
        if slot is None:
            possible = list(range(__TEXTURE_BINDING_OFFSET__, _maximum_tex_image_units()+1))
            for _, (_, order) in self.textures.items():
                possible.remove(order)
            if not possible:
                raise RuntimeError(f'No more texture slot available for material {self._name}')
            slot = possible[0]
            
        if slot in range(__TEXTURE_BINDING_OFFSET__):
            if is_dev_mode():
                EngineLogger.warning(f'Texture slot {slot} is reserved for FBO. Please use slot >= {__TEXTURE_BINDING_OFFSET__} instead.')
        for other_name, (_, other_order) in self.textures.items():
            if slot == other_order:
                if not replace:
                    raise RuntimeError(f'Order {slot} already exists in material {self._name}. Please remove it first.')
                else:
                    self.textures.pop(other_name)
                    break
        self.textures[name] = (texture, slot)

    def removeTexture(self, name:str)->Union[Texture, "CorrespondMap", None]:
        '''Remove texture by name. Return the removed texture(if not exists, return None)'''
        t = self.textures.pop(name, None)
        if t is None:
            return None
        return t[0]
        
    def addDefaultTexture(self, 
                          texture:Union[Texture, "CorrespondMap"],
                          default_type:DefaultTextureType,
                          slot: int|None=None):
        '''
        Add default texture with default name and order. Will replace the existing one with the same order & name.
        For cases that u adding the correspondence map as a sampler2DArray, the `default_type` must be `CorrespondMap`.
        '''
        from ..corrmap import CorrespondMap
        if isinstance(texture, CorrespondMap):
            if default_type != DefaultTextureType.CorrespondMap:
                raise TypeError(f'Unsupported default type {type(default_type)} for correspondence map. Supported types are {get_args(DefaultVariableType)}')
        elif default_type == DefaultTextureType.CorrespondMap:
            raise TypeError(f'Unsupported texture type {type(texture)} for default type {default_type}. Supported types are {Texture}, {CorrespondMap}')
        self.addTexture(default_type.value.name, texture, slot=slot, replace=True)
    
    def hasDefaultTexture(self, textureType:DefaultTextureType):
        '''Return True if the material has the default texture'''
        return textureType.value.name in self.textures
    
    def setVariable(self, name:str, value:SupportedShaderValueTypes):
        if not valueTypeCheck(value, SupportedShaderValueTypes):    # type: ignore
            raise TypeError(f'Unsupported value type {type(value)} for material variable {name}. Supported types are {get_args(SupportedShaderValueTypes)}')
        self.variables[name] = value
    
    def use(self):
        '''Use shader, and bind all textures to shader'''
        self.shader.useProgram()
        for name, (tex, slot) in self.textures.items():
            textureType = DefaultTextureType.FindDefaultTexture(name)
            if textureType is not None:
                # means this is a default texture
                self.shader.setUniform(textureType.value.shader_check_name, 1)  # e.g. hasDiffuseTex=1
            tex.bind(slot, self.shader.getUniformID(name))
            self.engine.CatchOpenGLError()
            
        for default_tex_type in DefaultTextureType.__members__.values():
            if default_tex_type.value.name not in self.textures:
                self.shader.setUniform(default_tex_type.value.shader_check_name, 0)
        
        for name, value in self.variables.items():
            self.shader.setUniform(name, value)
        
        self.shader.setUniform('materialID', self.materialID)   # for AI rendering to do tracing
        
    def unbind(self):
        '''Unbind all textures'''
        for _, (tex, slot) in self.textures.items():
            tex.unbind(slot)
        
        

__all__ = ['Material', 'DefaultTextureType']