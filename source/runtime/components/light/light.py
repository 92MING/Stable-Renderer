from runtime.component import Component, ComponentMeta
from static import Color
from typing import Literal
from utils.global_utils import GetGlobalValue, HasGlobalValue, GetOrAddGlobalValue
import OpenGL.GL as gl
import glm

_LightSubTypes = GetOrAddGlobalValue('_LightSubTypes', {}) # class name -> class
class LightMeta(ComponentMeta):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        clsName = obj.__qualname__
        if clsName != 'Light' and clsName not in _LightSubTypes:
            _LightSubTypes[clsName] = obj
        return obj

class LightShaderStruct:
    '''
    You can inherit this class to specify the struct of each light type in shader.
    Note that each attribute must be annotated with its type(e.g. `position: glm.vec3`).
    Also, types of attributes must be only one of the following types:
    `int`, `float`, `bool`, `glm.vec2`, `glm.vec3`, `glm.vec4`, `glm.mat2`, `glm.mat3`, `glm.mat4`,
    `glm.mat2x3`, `glm.mat2x4`, `glm.mat3x2`, `glm.mat3x4`, `glm.mat4x2`, `glm.mat4x3`, `glm.uint32`,
    `glm.uvec2`, `glm.uvec3`, `glm.uvec4`, `glm.ivec2`, `glm.ivec3`, `glm.ivec4`.
    '''
    # basic info
    position: glm.vec3
    color: glm.vec3
    intensity: float
    # attenuation
    att_const: float
    att_lin: float
    att_quad: float
    @classmethod
    def size(cls):
        '''Return the size of this struct in GLSL.'''
        size = 0
        for attType in cls.__annotations__.values():
            if attType in [int, float, bool, glm.uint32]:
                size += 4
            elif attType in [glm.vec2, glm.uvec2, glm.ivec2]:
                size += 8
            elif attType in [glm.vec3, glm.uvec3, glm.ivec3]:
                size += 12
            elif attType in [glm.vec4, glm.uvec4, glm.ivec4, glm.mat2]:
                size += 16
            elif attType in [glm.mat2x3, glm.mat3x2]:
                size += 24
            elif attType in [glm.mat2x4, glm.mat4x2]:
                size += 32
            elif attType in [glm.mat3x4, glm.mat4x3]:
                size += 48
            elif attType in [glm.mat3]:
                size += 36
            elif attType in [glm.mat4]:
                size += 64
            else:
                raise TypeError(f'Unsupported type {attType}.')
        return size

class Light(Component, metaclass=LightMeta):
    '''The abstract class of all light components.'''

    # region class constants/ properties/ methods
    _Shadow_Map_Dimension:Literal[2, 3] = 2
    '''Override this property to set the dimension of shadow map.'''
    _Shader_UBO_Index_Order = 0
    '''
    The order of the light in shader UBO "Lights" block. 
    Override this property to set the order of the light in shader UBO "Lights" block.
    '''
    _Shader_Light_Struct :type = LightShaderStruct
    '''Give the class of the struct of this light type in shader.'''

    _Default_Max_Num = 128
    '''
    Real Max Num of each light type can be changed through setting global value `MAX_NUM_{LightType.__qualname__}`. 
    But it cant be changed during runtime.
    '''
    @classmethod
    def Max_Num(cls):
        if HasGlobalValue(f'MAX_NUM_{cls.__qualname__}'):
            return GetGlobalValue(f'MAX_NUM_{cls.__qualname__}')
        return cls._Default_Max_Num # default value

    @staticmethod
    def AllLightSubTypes():
        '''Return all subtypes of Light, e.g. PointLight, DirLight, etc.'''
        return _LightSubTypes.values()

    _All_Lights = set()
    '''All lights in the scene.'''
    _All_Shadow_Lights = set()
    '''All lights that cast shadow in the scene.'''
    @staticmethod
    def AllLights(activeOnly=True):
        '''Return all lights instances in the scene.'''
        for light in Light._All_Lights:
            if activeOnly and not light.enable:
                continue
            yield light
    @staticmethod
    def AllShadowLights(activeOnly=True):
        '''Return all lights instances that cast shadow in the scene.'''
        for light in Light._All_Shadow_Lights:
            if activeOnly and not light.enable:
                continue
            yield light
    # endregion

    def __init__(self,
                 gameObj,
                 enable=True,
                 lightCol:Color = Color.WHITE,
                 intensity:float = 1.0,
                 att_const:float = 1.0, # attenuation constant
                 att_lin:float = 0.09, # attenuation linear
                 att_quad:float = 0.032, # attenuation quadratic
                 castShadow:bool = False, # if create shadow map
                 shadowMapSize:int = 512, # shadow map edge length if castShadow is True
                 ):
        if self.__class__.__qualname__ == 'Light':
            raise RuntimeError('Cannot instantiate abstract class Light.')
        super().__init__(gameObj, enable)
        self._lightCol = lightCol
        self._intensity = intensity
        self._att_const = att_const
        self._att_lin = att_lin
        self._att_quad = att_quad
        self._castShadow = castShadow
        self._shadowMapID = None
        self._shadowMapSize = shadowMapSize
        if castShadow:
            self._shadowMapID = self.engine.RenderManager.GetLightShadowMap(size=shadowMapSize, dimension=self._Shadow_Map_Dimension)
            self.__class__._All_Shadow_Lights.add(self)

    # region runtime
    def onDestroy(self):
        self.__class__._All_Lights.remove(self)
        if self._castShadow:
            self.__class__._All_Shadow_Lights.remove(self)
    # endregion

    # region properties
    @property
    def shadowMapID(self):
        return self._shadowMapID
    @property
    def shadowMapSize(self):
        return self._shadowMapSize
    @shadowMapSize.setter
    def shadowMapSize(self, size:int):
        '''Warning: this may trigger the creation of shadow map if the target size is not created in the system.'''
        if self._shadowMapSize == size:
            return
        self._shadowMapSize = size
        if self._castShadow:
            self._tryDeleteShadowMap()
            self._shadowMapID = self.engine.RenderManager.GetLightShadowMap(size=size, dimension=self._Shadow_Map_Dimension)
    @property
    def lightColor(self):
        return self._lightCol
    @lightColor.setter
    def lightColor(self, value):
        col = Color(*value)
        self._lightCol = col
    @property
    def intensity(self)->float:
        return self._intensity
    @intensity.setter
    def intensity(self, value:float):
        self._intensity = float(value)
    @property
    def attenuationConstant(self)->float:
        return self._att_const
    @attenuationConstant.setter
    def attenuationConstant(self, value:float):
        self._att_const = value
    @property
    def attenuationLinear(self)->float:
        return self._att_lin
    @attenuationLinear.setter
    def attenuationLinear(self, value:float):
        self._att_lin = value
    @property
    def attenuationQuadratic(self)->float:
        return self._att_quad
    @attenuationQuadratic.setter
    def attenuationQuadratic(self, value:float):
        self._att_quad = value
    @property
    def castShadow(self)->bool:
        return self._castShadow
    @castShadow.setter
    def castShadow(self, value:bool):
        '''If you want to specify the shadow map size, please use "setCastShadow"'''
        self.setCastShadow(value)
    def setCastShadow(self, set:bool, shadowMapSize=None):
        '''Warning: Not recommended to change this property during runtime since it may create shadow map.'''
        if set == self._castShadow:
            return
        if shadowMapSize is None:
            shadowMapSize = self._shadowMapSize
        if set:
            self._shadowMapID = self.engine.RenderManager.GetLightShadowMap(size=shadowMapSize, dimension=self._Shadow_Map_Dimension)
            self.__class__._All_Shadow_Lights.add(self)
        else:
            self._tryDeleteShadowMap() # try delete shadow map if no more other lights use it
            self.__class__._All_Shadow_Lights.remove(self)
    # endregion

    # region private
    def _tryDeleteShadowMap(self):
        '''try delete shadow map if no more other lights use it'''
        if self._shadowMapID is None:
            return # no shadow map
        shouldDel = True
        for otherShadowLight in Light.AllShadowLights(activeOnly=False):
            if otherShadowLight._shadowMapID == self._shadowMapID:
                shouldDel = False
                break
        if shouldDel:
            self.engine.RenderManager.DeleteLightShadowMap(self._shadowMapID)
            self._shadowMapID = None
    def _drawShadow(self):
        if not self.castShadow:
            return
        renderManager = self.engine.RenderManager
        renderManager.BindFrameBuffer(renderManager.LightShadowFBO)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        if self._Shadow_Map_Dimension == 2:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._shadowMapID, 0)
            gl.glViewport(0, 0, self._shadowMapSize, self._shadowMapSize)

        else:
            pass
    # endregion


__all__ = ['Light', 'LightShaderStruct', 'LightMeta']