# -*- coding: utf-8 -*-
'''this module contains enums, constants, etc.'''
import glm
import torch
import numpy as np
import OpenGL.GL as gl
import cupy

from typing import Optional
from enum import Enum
from OpenGL.constant import Constant as GL_Constant
from dataclasses import dataclass


# region engine
class EngineMode(Enum):
    GAME = 0
    '''only the rendering window (glfw) will be created.'''
    EDITOR = 1
    '''the UI editor(PYQT) will be opened. ComfyUI will also be started.'''
    BAKE = 2
    '''Baking mode is for baking the StableDiffusion's rendering result into a pack of textures. ComfyUI will also be forced to start in this mode.'''

class EngineStage(Enum):
    '''
    The stage of engine. It will be changed during the engine running.
    When u set the property `Stage` in engine, it will invoke the event `_OnEngineStageChanged`.
    '''
    
    INIT = 0
    
    BEFORE_PREPARE = 1
    AFTER_PREPARE = 2
    
    BEFORE_FRAME_BEGIN = 3
    BEFORE_FRAME_RUN = 4
    BEFORE_FRAME_END = 5
    
    BEFORE_RELEASE = 6
    
    BEFORE_PAUSE = 7
    PAUSE = 8
    
    ENDED = 9
    
    @staticmethod
    def PreparingStages():
        return (EngineStage.BEFORE_PREPARE, EngineStage.AFTER_PREPARE)
    
    @staticmethod
    def RunningStages():
        return (EngineStage.BEFORE_FRAME_BEGIN, EngineStage.BEFORE_FRAME_RUN, EngineStage.BEFORE_FRAME_END )

class EngineFBO(Enum):
    '''
    Bind GBuffer textures to the target shader.
    Please ensure that your shader has the following uniforms:
        
    - currentColor: sampler2D (vec4)
    - currentIDs: usampler2D (ivec4)
    - currentPos: sampler2D (vec3)
    - currentNormalDepth: sampler2D (vec4)
    - currentNoises: sampler2D  (vec4)
    
    value=(binding point, uniform name)
    '''
    COLOR = (0, 'currentColor')
    '''Color buffer, rgba, f16'''
    ID = (1, 'currentIDs')
    '''ID buffer, (spriteID, materialID, map_index, vertexID), uint32(but since torch does not support uint32, it will be converted to int32 later'''
    POS = (2, 'currentPos')
    '''Position buffer'''
    NORMAL_DEPTH = (3, 'currentNormalDepth')
    '''Normal and Depth buffer. (nx, ny, nz, depth), float16'''
    NOISE = (4, 'currentNoises')
    '''Latent. 4 channels, float16'''
    CANNY = (5, 'currentCanny')
    '''Canny edge detection. 1 channel, float16'''
    
    @property
    def binding_pt(self)->int:
        return self.value[0]
    
    @property
    def uniform_name(self)->str:
        return self.value[1] 

@dataclass
class DefaultTexture:
    '''This class is used to store the default texture info in glsl.'''
    name: str
    '''Name of the default texture, e.g. diffuseTex'''
    shader_check_name: str
    '''for glsl internal use, e.g. "hasDiffuseTex"'''

class DefaultTextureType(Enum):
    '''This class is used to store the default texture types. e.g. DiffuseTex, NormalTex, etc.'''

    DiffuseTex = DefaultTexture('diffuseTex', 'hasDiffuseTex')
    '''Color texture of the material'''
    NormalTex = DefaultTexture('normalTex', 'hasNormalTex')
    '''Normal texture of the material'''
    SpecularTex = DefaultTexture('specularTex', 'hasSpecularTex')
    '''Light reflection texture of the material'''
    EmissionTex = DefaultTexture('emissionTex', 'hasEmissionTex')
    '''Light emission texture of the material'''
    OcclusionTex = DefaultTexture('occlusionTex', 'hasOcclusionTex')
    '''Ambient occlusion texture of the material. Known as `AO` texture.'''
    MetallicTex = DefaultTexture('metallicTex', 'hasMetallicTex')
    '''Metallic texture of the material'''
    RoughnessTex = DefaultTexture('roughnessTex', 'hasRoughnessTex')
    '''Roughness texture of the material'''
    DisplacementTex = DefaultTexture('displacementTex', 'hasDisplacementTex')
    '''Displacement texture of the material'''
    AlphaTex = DefaultTexture('alphaTex', 'hasAlphaTex')
    '''Alpha texture of the material'''
    NoiseTex = DefaultTexture('noiseTex', 'hasNoiseTex')
    '''Noise texture, for AI rendering'''
    CorrespondMap = DefaultTexture('correspond_map', 'hasCorrMap')
    '''correspondence map, for AI baked rendering'''

    @classmethod
    def FindDefaultTexture(cls, texName:str):
        '''Return DefaultTextureType with given name. Return None if not found.'''
        texName = texName.lower()
        for item in cls:
            if item.value.name.lower() == texName:
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


__all__ = ['EngineMode', 'EngineStage', 'EngineFBO', 'DefaultTextureType', 'DefaultTexture', 'DefaultVariableType', 'DefaultVariable']
# endregion

# region render
class DepthFunc(Enum):
    '''The comparison function for depth test. See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDepthFunc.xhtml'''
    
    NEVER = gl.GL_NEVER
    LESS = gl.GL_LESS
    EQUAL = gl.GL_EQUAL
    LESS_EQUAL = gl.GL_LEQUAL
    GREATER = gl.GL_GREATER
    NOT_EQUAL = gl.GL_NOTEQUAL
    GREATER_EQUAL = gl.GL_GEQUAL
    ALWAYS = gl.GL_ALWAYS

class RenderOrder(Enum):
    """Render order of GameObjects"""
    
    OPAQUE = 1000
    TRANSPARENT = 2000
    OVERLAY = 3000
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        elif isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        elif isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        elif isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        elif isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented
    
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        elif isinstance(other, (int, float)):
            return self.value == other
        return NotImplemented
    
    def __ne__(self, other):
        if self.__class__ is other.__class__:
            return self.value != other.value
        elif isinstance(other, (int, float)):
            return self.value != other
        return NotImplemented
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.value + other
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.value - other
        return NotImplemented
    
class RenderMode(Enum):
    NORMAL = 0
    '''normal object'''
    BAKED = 1
    '''baked object getting color from corrmap'''
    BAKING = 2
    '''baking object to corrmap'''
    
__all__.extend(['RenderOrder', 'RenderMode', 'DepthFunc'])
# endregion


# region camera
class ProjectionType(Enum):
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1

__all__.extend(['ProjectionType'])
# endregion


# region light
class LightType(Enum):
    DIRECTIONAL_LIGHT = 0
    POINT_LIGHT = 1
    SPOT_LIGHT = 2

__all__.extend(['LightType'])
# endregion


# region input
class _FindableEnum(Enum):
    @classmethod
    def GetEnum(cls, value):
        for e in cls:
            if e.value == value:
                return e
        return None

class MouseButton(_FindableEnum):
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2
    MOUSE_BTN_3 = 3
    MOUSE_BTN_4 = 4
    MOUSE_BTN_5 = 5
    MOUSE_BTN_6 = 6
    MOUSE_BTN_7 = 7
    MOUSE_BTN_8 = 8

class InputAction(_FindableEnum):
    RELEASE = 0
    PRESS = 1
    HOLD = 999

class InputModifier(_FindableEnum):
    '''Keyboard input Modifier'''
    NONE = 0
    SHIFT = 1
    CTRL = 2
    ALT = 4
    SUPER = 8

class GLFW_Key(_FindableEnum):
    '''Keyboard codes of GLFW'''

    UNKNOWN = -1
    SPACE = 32
    APOSTROPHE = 39
    COMMA = 44
    MINUS = 45
    PERIOD = 46
    SLASH = 47
    KEY_0 = 48
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_5 = 53
    KEY_6 = 54
    KEY_7 = 55
    KEY_8 = 56
    KEY_9 = 57
    SEMICOLON = 59
    EQUAL = 61
    A = 65
    B = 66
    C = 67
    D = 68
    E = 69
    F = 70
    G = 71
    H = 72
    I = 73
    J = 74
    K = 75
    L = 76
    M = 77
    N = 78
    O = 79
    P = 80
    Q = 81
    R = 82
    S = 83
    T = 84
    U = 85
    V = 86
    W = 87
    X = 88
    Y = 89
    Z = 90
    LEFT_BRACKET = 91
    BACKSLASH = 92
    RIGHT_BRACKET = 93
    GRAVE_ACCENT = 96
    WORLD_1 = 161
    WORLD_2 = 162
    ESCAPE = 256
    ENTER = 257
    TAB = 258
    BACKSPACE = 259
    INSERT = 260
    DELETE = 261
    RIGHT = 262
    LEFT = 263
    DOWN = 264
    UP = 265
    PAGE_UP = 266
    PAGE_DOWN = 267
    HOME = 268
    END = 269
    CAPS_LOCK = 280
    SCROLL_LOCK = 281
    NUM_LOCK = 282
    PRINT_SCREEN = 283
    PAUSE = 284
    F1 = 290
    F2 = 291
    F3 = 292
    F4 = 293
    F5 = 294
    F6 = 295
    F7 = 296
    F8 = 297
    F9 = 298
    F10 = 299
    F11 = 300
    F12 = 301

__all__.extend(['MouseButton', 'InputAction', 'InputModifier', 'GLFW_Key'])
# endregion


# region opengl
class TextureWrap(Enum):
    REPEAT = gl.GL_REPEAT
    MIRRORED_REPEAT = gl.GL_MIRRORED_REPEAT
    MIRROR_CLAMP_TO_EDGE = gl.GL_MIRROR_CLAMP_TO_EDGE
    CLAMP_TO_EDGE = gl.GL_CLAMP_TO_EDGE
    CLAMP_TO_BORDER = gl.GL_CLAMP_TO_BORDER

class TextureFilter(Enum):
    NEAREST = gl.GL_NEAREST
    LINEAR = gl.GL_LINEAR
    NEAREST_MIPMAP_NEAREST = gl.GL_NEAREST_MIPMAP_NEAREST
    LINEAR_MIPMAP_NEAREST = gl.GL_LINEAR_MIPMAP_NEAREST
    NEAREST_MIPMAP_LINEAR = gl.GL_NEAREST_MIPMAP_LINEAR
    LINEAR_MIPMAP_LINEAR = gl.GL_LINEAR_MIPMAP_LINEAR

@dataclass
class TextureDataTypeItem:
    '''Enum value for `TextureDataType`'''
    
    gl_data_type : GL_Constant
    '''OpenGL data type enum'''
    numpy_dtype : type
    '''numpy data type'''
    
    @property
    def nbytes(self)->int:
        '''return the number of bytes for the data type'''
        return np.dtype(self.numpy_dtype).itemsize
    
    @property
    def tensor_supported(self)->bool:
        '''whether the data type is supported by torch tensor directly'''
        return self.torch_dtype is not None
    
    @property
    def torch_dtype(self)->Optional[torch.dtype]:
        '''
        The torch dtype for turning into tensor directly.
        
        Please be aware that not some data types are not directly supported by torch, 
        it will turn into another dtype with the same bit width.
        
        For types not supported by torch, it will return None.
        '''
        match self.numpy_dtype:
            case np.uint8:
                return torch.uint8
            case np.int8:
                return torch.int8
            
            case np.uint16:
                return torch.int16  # torch does not support uint16
            case np.int16:
                return torch.int16
            case np.float16:
                return torch.float16
            
            case np.uint32:
                return torch.int32  # torch does not support uint32
            case np.int32:
                return torch.int32
            case np.float32:
                return torch.float32
            
            case np.uint64:
                return torch.int64
            case np.int64:
                return torch.int64
            case np.float64:
                return torch.float64
            
            case _:
                return None
    
    @property
    def cupy_dtype(self):
        '''return the related dtype in cupy'''
        match self.numpy_dtype:
            case np.uint8:
                return cupy.uint8
            case np.int8:
                return cupy.int8
            
            case np.uint16:
                return cupy.uint16
            case np.int16:  
                return cupy.int16
            case np.float16:
                return cupy.float16
            
            case np.uint32:
                return cupy.uint32
            case np.int32:
                return cupy.int32
            case np.float32:
                return cupy.float32
            
            case np.uint64:
                return cupy.uint64
            case np.int64:
                return cupy.int64
            case np.float64:
                return cupy.float64
            
            case _:
                raise ValueError(f'No matching cupy dtype for {self.numpy_dtype}')
    
class TextureDataType(Enum):
    '''
    Acceptable data type for sending texture to GPU.
    See: 
        https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        https://www.khronos.org/opengl/wiki/Pixel_Transfer#Pixel_type
    '''
    
    # integers
    UNSIGNED_BYTE = TextureDataTypeItem(gl.GL_UNSIGNED_BYTE, np.uint8)
    '''8 bits unsigned integer'''
    BYTE = TextureDataTypeItem(gl.GL_BYTE, np.int8)
    '''8 bits (signed) integer'''
    
    UNSIGNED_SHORT = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT, np.uint16)
    '''16 bits unsigned integer'''
    SHORT = TextureDataTypeItem(gl.GL_SHORT, np.int16)
    '''16 bits (signed) integer'''
    
    UNSIGNED_INT = TextureDataTypeItem(gl.GL_UNSIGNED_INT, np.uint32)
    '''32 bits unsigned integer'''
    INT = TextureDataTypeItem(gl.GL_INT, np.int32)
    '''32 bits (signed) integer'''
    
    # floating point
    HALF = TextureDataTypeItem(gl.GL_HALF_FLOAT, np.float16)
    '''float 16'''
    FLOAT = TextureDataTypeItem(gl.GL_FLOAT, np.float32)
    '''float 32'''

    # special data arrangement
    UNSIGNED_BYTE_332 = TextureDataTypeItem(gl.GL_UNSIGNED_BYTE_3_3_2, np.uint8)
    UNSIGNED_BYTE_233 = TextureDataTypeItem(gl.GL_UNSIGNED_BYTE_2_3_3_REV, np.uint8)
    
    UNSIGNED_SHORT_565 = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_5_6_5, np.uint16)
    '''Color saving order: BGR, 5 bits for blue, 6 bits for green, 5 bits for red.'''
    UNSIGNED_SHORT_565_REV = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_5_6_5_REV, np.uint16)
    '''Color saving order: RGB, 5 bits for red, 6 bits for green, 5 bits for blue.'''
    
    UNSIGNED_SHORT_4444 = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_4_4_4_4, np.uint16)
    '''Color saving order: RGBA, 4 bits for red, 4 bits for green, 4 bits for blue, 4 bits for alpha.'''
    UNSIGNED_SHORT_4444_REV = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_4_4_4_4_REV, np.uint16)
    '''Color saving order: ARGB, 4 bits for alpha, 4 bits for red, 4 bits for green, 4 bits for blue.'''
    
    UNSIGNED_SHORT_5551 = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_5_5_5_1, np.uint16)
    '''Color saving order: RGBA, 5 bits for red, 5 bits for green, 5 bits for blue, 1 bit for alpha.'''
    UNSIGNED_SHORT_1555_REV = TextureDataTypeItem(gl.GL_UNSIGNED_SHORT_1_5_5_5_REV, np.uint16)
    '''Color saving order: ARGB, 1 bit for alpha, 5 bits for red, 5 bits for green, 5 bits for blue.'''
    
    UNSIGNED_INT_8888 = TextureDataTypeItem(gl.GL_UNSIGNED_INT_8_8_8_8, np.uint32)
    '''Color saving order: RGBA, 8 bits for red, 8 bits for green, 8 bits for blue, 8 bits for alpha.'''
    UNSIGNED_INT_8888_REV = TextureDataTypeItem(gl.GL_UNSIGNED_INT_8_8_8_8_REV, np.uint32)
    '''Color saving order: ARGB, 8 bits for alpha, 8 bits for red, 8 bits for green, 8 bits for blue.'''
    
    UNSIGNED_INT_10_10_10_2 = TextureDataTypeItem(gl.GL_UNSIGNED_INT_10_10_10_2, np.uint32)
    '''Color saving order: RGBA, 10 bits for red, 10 bits for green, 10 bits for blue, 2 bits for alpha.'''
    UNSIGNED_INT_2_10_10_10_REV = TextureDataTypeItem(gl.GL_UNSIGNED_INT_2_10_10_10_REV, np.uint32)
    '''Color saving order: ARGB, 2 bits for alpha, 10 bits for red, 10 bits for green, 10 bits for blue.'''
    
    UNSIGNED_INT_24_8 = TextureDataTypeItem(gl.GL_UNSIGNED_INT_24_8, np.uint32)
    '''24 bits for depth, 8 bits for stencil.'''
    UNSIGNED_INT_10F_11F_11F_REV = TextureDataTypeItem(gl.GL_UNSIGNED_INT_10F_11F_11F_REV, np.uint32)
    '''11 bits for red, 11 bits for green, 10 bits for blue, and the data is floating point.'''
    UNSIGNED_INT_5_9_9_9_REV = TextureDataTypeItem(gl.GL_UNSIGNED_INT_5_9_9_9_REV, np.uint32)
    '''9 bits for red, 9 bits for green, 9 bits for blue, and the data is floating point.'''
    FLOAT_32_UNSIGNED_INT_24_8_REV = TextureDataTypeItem(gl.GL_FLOAT_32_UNSIGNED_INT_24_8_REV, np.uint64)    # uint64 for same size only
    '''
    The first value is a 32-bit floating-point depth value. The second breaks the 32-bit integer value into 24-bits of unused space, followed by 8 bits of stencil.
    This can only be used with `GL_DEPTH32F_STENCIL8`
    '''

@dataclass
class TextureInternalFormatItem:
    '''
    Dataclass for `TextureInternalFormat` to specify the real GL enum and the recommended gl data type.
    
    Difference of internal format and format:
        - format: specifies the channel order/ number
        - internal format: specifies the data type/size
    '''
    gl_internal_format : GL_Constant
    '''OpenGL internal format enum'''
    default_data_type : TextureDataType
    '''The recommended gl data type for this internal format.'''
    @property
    def default_gl_data_type(self)->GL_Constant:
        return self.default_data_type.value.gl_data_type

class TextureInternalFormat(Enum):
    '''
    Internal texture format for opengl.
    
    See: 
        https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        https://www.khronos.org/opengl/wiki/Pixel_Transfer#Pixel_type
    '''
    
    # 1: Base Internal Formats
    RED = TextureInternalFormatItem(gl.GL_RED, TextureDataType.UNSIGNED_BYTE)
    RG = TextureInternalFormatItem(gl.GL_RG, TextureDataType.UNSIGNED_BYTE)
    RGB = TextureInternalFormatItem(gl.GL_RGB, TextureDataType.UNSIGNED_BYTE)
    RGBA = TextureInternalFormatItem(gl.GL_RGBA, TextureDataType.UNSIGNED_BYTE)
    
    DEPTH = TextureInternalFormatItem(gl.GL_DEPTH_COMPONENT, TextureDataType.FLOAT)
    DEPTH_STENCIL = TextureInternalFormatItem(gl.GL_DEPTH_STENCIL, TextureDataType.UNSIGNED_INT_24_8)
    DEPTH_16 = TextureInternalFormatItem(gl.GL_DEPTH_COMPONENT16, TextureDataType.UNSIGNED_SHORT)
    DEPTH_24 = TextureInternalFormatItem(gl.GL_DEPTH_COMPONENT24, TextureDataType.UNSIGNED_INT)
    DEPTH_32 = TextureInternalFormatItem(gl.GL_DEPTH_COMPONENT32, TextureDataType.UNSIGNED_INT)
    DEPTH_32F = TextureInternalFormatItem(gl.GL_DEPTH_COMPONENT32F, TextureDataType.FLOAT)
    DEPTH_24_STENCIL_8 = TextureInternalFormatItem(gl.GL_DEPTH24_STENCIL8, TextureDataType.UNSIGNED_INT_24_8)
    DEPTH_32F_STENCIL_8 = TextureInternalFormatItem(gl.GL_DEPTH32F_STENCIL8, TextureDataType.FLOAT_32_UNSIGNED_INT_24_8_REV)
    
    STENCIL_INDEX = TextureInternalFormatItem(gl.GL_STENCIL_INDEX, TextureDataType.UNSIGNED_BYTE)
    STENCIL_INDEX_1 = TextureInternalFormatItem(gl.GL_STENCIL_INDEX1, TextureDataType.UNSIGNED_BYTE)
    STENCIL_INDEX_4 = TextureInternalFormatItem(gl.GL_STENCIL_INDEX4, TextureDataType.UNSIGNED_BYTE)
    STENCIL_INDEX_8 = TextureInternalFormatItem(gl.GL_STENCIL_INDEX8, TextureDataType.UNSIGNED_BYTE)
    STENCIL_INDEX_16 = TextureInternalFormatItem(gl.GL_STENCIL_INDEX16, TextureDataType.UNSIGNED_SHORT)
    
    # 2: Sized Internal Formats
    RED_8 = TextureInternalFormatItem(gl.GL_RG8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RED` but with 8 bits per channel.'''
    RED_8_SNORM = TextureInternalFormatItem(gl.GL_R8_SNORM, TextureDataType.BYTE)
    '''same with `RED` but with 8 bits per channel, and the data is normalized to [-1, 1].'''
    RED_16 = TextureInternalFormatItem(gl.GL_R16, TextureDataType.UNSIGNED_SHORT)
    '''same with `RED` but with 16 bits per channel.'''
    RED_16_SNORM = TextureInternalFormatItem(gl.GL_R16_SNORM, TextureDataType.SHORT)
    '''same with `RED` but with 16 bits per channel, and the data is normalized to [-1, 1].'''
    RG_8 = TextureInternalFormatItem(gl.GL_RG8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RG` but with 8 bits per channel.'''
    RG_8_SNORM = TextureInternalFormatItem(gl.GL_RG8_SNORM, TextureDataType.BYTE)
    '''same with `RG` but with 8 bits per channel, and the data is normalized to [-1, 1].'''
    RG_16 = TextureInternalFormatItem(gl.GL_RG16, TextureDataType.UNSIGNED_SHORT)
    '''same with `RG` but with 16 bits per channel.'''
    RG_16_SNORM = TextureInternalFormatItem(gl.GL_RG16_SNORM, TextureDataType.SHORT)
    '''same with `RG` but with 16 bits per channel, and the data is normalized to [-1, 1].'''
    R3_G3_B2 = TextureInternalFormatItem(gl.GL_R3_G3_B2, TextureDataType.UNSIGNED_BYTE_233)
    '''same with `RGB` but with 3 bits for red, 3 bits for green, 2 bits for blue.'''
    RGB_4 = TextureInternalFormatItem(gl.GL_RGB4, TextureDataType.UNSIGNED_SHORT_565)
    '''same with `RGB` but with 4 bits per channel.'''
    RGB_5 = TextureInternalFormatItem(gl.GL_RGB5, TextureDataType.UNSIGNED_SHORT_565)
    '''same with `RGB` but with 5 bits per channel.'''
    RGB_8 = TextureInternalFormatItem(gl.GL_RGB8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGB` but with 8 bits per channel.'''
    RGB_8_SNORM = TextureInternalFormatItem(gl.GL_RGB8_SNORM, TextureDataType.BYTE)
    '''same with `RGB` but with 8 bits per channel, and the data is normalized to [-1, 1].'''
    RGB_10 = TextureInternalFormatItem(gl.GL_RGB10, TextureDataType.UNSIGNED_INT_10_10_10_2)
    '''same with `RGB` but with 10 bits per channel.'''
    RGB_12 = TextureInternalFormatItem(gl.GL_RGB12, TextureDataType.UNSIGNED_SHORT)
    '''same with `RGB` but with 12 bits per channel.'''
    RGB_16_SNORM = TextureInternalFormatItem(gl.GL_RGB16_SNORM, TextureDataType.SHORT)
    '''same with `RGB` but with 16 bits per channel, and the data is normalized to [-1, 1].'''
    RGBA_2 = TextureInternalFormatItem(gl.GL_RGBA2, TextureDataType.UNSIGNED_BYTE_233)
    '''same with `RGBA` but with 2 bits per channel.'''
    RGBA_4 = TextureInternalFormatItem(gl.GL_RGBA4, TextureDataType.UNSIGNED_SHORT_4444)
    '''same with `RGBA` but with 4 bits per channel.'''
    RGB5_A1 = TextureInternalFormatItem(gl.GL_RGB5_A1, TextureDataType.UNSIGNED_SHORT_5551)
    '''same with `RGBA` but with 5 bits for red, 5 bits for green, 5 bits for blue, 1 bit for alpha.'''
    RGBA_8 = TextureInternalFormatItem(gl.GL_RGBA8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with 8 bits per channel.'''
    RGBA_8_SNORM = TextureInternalFormatItem(gl.GL_RGBA8_SNORM, TextureDataType.BYTE)
    '''same with `RGBA` but with 8 bits per channel, and the data is normalized to [-1, 1].'''
    RGB10_A2 = TextureInternalFormatItem(gl.GL_RGB10_A2, TextureDataType.UNSIGNED_INT_10_10_10_2)
    '''same with `RGBA` but with 10 bits for red, 10 bits for green, 10 bits for blue, 2 bits for alpha.'''
    RGB10_A2UI = TextureInternalFormatItem(gl.GL_RGB10_A2UI, TextureDataType.UNSIGNED_INT_10_10_10_2)
    '''same with `RGBA` but with 10 bits for red, 10 bits for green, 10 bits for blue, 2 bits for alpha, and the data is unsigned integer.'''
    RGBA12 = TextureInternalFormatItem(gl.GL_RGBA12, TextureDataType.UNSIGNED_SHORT)
    '''same with `RGBA` but with 12 bits per channel.'''
    RGBA16 = TextureInternalFormatItem(gl.GL_RGBA16, TextureDataType.UNSIGNED_SHORT)
    '''same with `RGBA` but with 16 bits per channel, in integer.'''
    SRGB8 = TextureInternalFormatItem(gl.GL_SRGB8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGB` but with 8 bits per channel, and the data is in sRGB color space.'''
    SRGB8_A8 = TextureInternalFormatItem(gl.GL_SRGB8_ALPHA8, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with 8 bits per channel, and the data is in sRGB color space.'''
    RED_16F = TextureInternalFormatItem(gl.GL_R16F, TextureDataType.HALF)
    '''same with `RED` but with 16 bits per channel, and the data is floating point.'''
    RG16F = TextureInternalFormatItem(gl.GL_RG16F, TextureDataType.HALF)
    '''same with `RG` but with 16 bits per channel, and the data is floating point.'''
    RGB16F = TextureInternalFormatItem(gl.GL_RGB16F, TextureDataType.HALF)
    '''same with `RGB` but with 16 bits per channel, and the data is floating point.'''
    RGBA16F = TextureInternalFormatItem(gl.GL_RGBA16F, TextureDataType.HALF)
    '''same with `RGBA` but with 16 bits per channel, and the data is floating point.'''
    RED_32F = TextureInternalFormatItem(gl.GL_R32F, TextureDataType.FLOAT)
    '''same with `RED` but with 32 bits per channel, and the data is floating point.'''
    RG32F = TextureInternalFormatItem(gl.GL_RG32F, TextureDataType.FLOAT)
    '''same with `RG` but with 32 bits per channel, and the data is floating point.'''
    RGB32F = TextureInternalFormatItem(gl.GL_RGB32F, TextureDataType.FLOAT)
    '''same with `RGB` but with 32 bits per channel, and the data is floating point.'''
    RGBA32F = TextureInternalFormatItem(gl.GL_RGBA32F, TextureDataType.FLOAT)
    '''same with `RGBA` but with 32 bits per channel, and the data is floating point.'''
    R11F_G11F_B10F = TextureInternalFormatItem(gl.GL_R11F_G11F_B10F, TextureDataType.UNSIGNED_INT_10F_11F_11F_REV)
    '''same with `RGB` but with 11 bits for red, 11 bits for green, 10 bits for blue, and the data is floating point.'''
    RGB9_E5 = TextureInternalFormatItem(gl.GL_RGB9_E5, TextureDataType.UNSIGNED_INT_5_9_9_9_REV)
    '''same with `RGB` but with 9 bits for red, 9 bits for green, 9 bits for blue, and 5 bits for exponent, and the data is floating point.'''
    RED_8I = TextureInternalFormatItem(gl.GL_R8I, TextureDataType.BYTE)
    '''same with `RED` but with 8 bits per channel, and the data is signed integer.'''
    RED_8UI = TextureInternalFormatItem(gl.GL_R8UI, TextureDataType.UNSIGNED_BYTE)
    '''same with `RED` but with 8 bits per channel, and the data is unsigned integer.'''
    RED_16I = TextureInternalFormatItem(gl.GL_R16I, TextureDataType.SHORT)
    '''same with `RED` but with 16 bits per channel, and the data is signed integer.'''
    RED_16UI = TextureInternalFormatItem(gl.GL_R16UI, TextureDataType.UNSIGNED_SHORT)
    '''same with `RED` but with 16 bits per channel, and the data is unsigned integer.'''
    RED_32I = TextureInternalFormatItem(gl.GL_R32I, TextureDataType.INT)
    '''same with `RED` but with 32 bits per channel, and the data is signed integer.'''
    RED_32UI = TextureInternalFormatItem(gl.GL_R32UI, TextureDataType.UNSIGNED_INT)
    '''same with `RED` but with 32 bits per channel, and the data is unsigned integer.'''
    RG_8I = TextureInternalFormatItem(gl.GL_RG8I, TextureDataType.BYTE)
    '''same with `RG` but with 8 bits per channel, and the data is signed integer.'''
    RG_8UI = TextureInternalFormatItem(gl.GL_RG8UI, TextureDataType.UNSIGNED_BYTE)
    '''same with `RG` but with 8 bits per channel, and the data is unsigned integer.'''
    RG_16I = TextureInternalFormatItem(gl.GL_RG16I, TextureDataType.SHORT)
    '''same with `RG` but with 16 bits per channel, and the data is signed integer.'''
    RG_16UI = TextureInternalFormatItem(gl.GL_RG16UI, TextureDataType.UNSIGNED_SHORT)
    '''same with `RG` but with 16 bits per channel, and the data is unsigned integer.'''
    RG_32I = TextureInternalFormatItem(gl.GL_RG32I, TextureDataType.INT)
    '''same with `RG` but with 32 bits per channel, and the data is signed integer.'''
    RG_32UI = TextureInternalFormatItem(gl.GL_RG32UI, TextureDataType.UNSIGNED_INT)
    '''same with `RG` but with 32 bits per channel, and the data is unsigned integer.'''
    RGB_8I = TextureInternalFormatItem(gl.GL_RGB8I, TextureDataType.BYTE)
    '''same with `RGB` but with 8 bits per channel, and the data is signed integer.'''
    RGB_8UI = TextureInternalFormatItem(gl.GL_RGB8UI, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGB` but with 8 bits per channel, and the data is unsigned integer.'''
    RGB_16I = TextureInternalFormatItem(gl.GL_RGB16I, TextureDataType.SHORT)
    '''same with `RGB` but with 16 bits per channel, and the data is signed integer.'''
    RGB_16UI = TextureInternalFormatItem(gl.GL_RGB16UI, TextureDataType.UNSIGNED_SHORT)
    '''same with `RGB` but with 16 bits per channel, and the data is unsigned integer.'''
    RGB_32I = TextureInternalFormatItem(gl.GL_RGB32I, TextureDataType.INT)
    '''same with `RGB` but with 32 bits per channel, and the data is signed integer.'''
    RGB_32UI = TextureInternalFormatItem(gl.GL_RGB32UI, TextureDataType.UNSIGNED_INT)
    '''same with `RGB` but with 32 bits per channel, and the data is unsigned integer.'''
    RGBA_8I = TextureInternalFormatItem(gl.GL_RGBA8I, TextureDataType.BYTE)
    '''same with `RGBA` but with 8 bits per channel, and the data is signed integer.'''
    RGBA_8UI = TextureInternalFormatItem(gl.GL_RGBA8UI, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with 8 bits per channel, and the data is unsigned integer.'''
    RGBA_16I = TextureInternalFormatItem(gl.GL_RGBA16I, TextureDataType.SHORT)
    '''same with `RGBA` but with 16 bits per channel, and the data is signed integer.'''
    RGBA_16UI = TextureInternalFormatItem(gl.GL_RGBA16UI, TextureDataType.UNSIGNED_SHORT)
    '''same with `RGBA` but with 16 bits per channel, and the data is unsigned integer.'''
    RGBA_32I = TextureInternalFormatItem(gl.GL_RGBA32I, TextureDataType.INT)
    '''same with `RGBA` but with 32 bits per channel, and the data is signed integer.'''
    RGBA_32UI = TextureInternalFormatItem(gl.GL_RGBA32UI, TextureDataType.UNSIGNED_INT)
    '''same with `RGBA` but with 32 bits per channel, and the data is unsigned integer.'''
    
    # 3: Compressed Internal Formats
    COMPRESSED_RED = TextureInternalFormatItem(gl.GL_COMPRESSED_RED, TextureDataType.UNSIGNED_BYTE)
    '''same with `RED` but with compressed data.'''
    COMPRESSED_RG = TextureInternalFormatItem(gl.GL_COMPRESSED_RG, TextureDataType.UNSIGNED_BYTE)
    '''same with `RG` but with compressed data.'''
    COMPRESSED_RGB = TextureInternalFormatItem(gl.GL_COMPRESSED_RGB, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGB` but with compressed data.'''
    COMPRESSED_RGBA = TextureInternalFormatItem(gl.GL_COMPRESSED_RGBA, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with compressed data.'''
    COMPRESSED_SRGB = TextureInternalFormatItem(gl.GL_COMPRESSED_SRGB, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGB` but with compressed data, and the data is in sRGB color space.'''
    COMPRESSED_SRGB_A = TextureInternalFormatItem(gl.GL_COMPRESSED_SRGB_ALPHA, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with compressed data, and the data is in sRGB color space.'''
    COMPRESSED_RED_RGTC1 = TextureInternalFormatItem(gl.GL_COMPRESSED_RED_RGTC1, TextureDataType.UNSIGNED_BYTE)
    '''same with `RED` but with compressed data, and the data is in RGTC1 format.'''
    COMPRESSED_SIGNED_RED_RGTC1 = TextureInternalFormatItem(gl.GL_COMPRESSED_SIGNED_RED_RGTC1, TextureDataType.BYTE)
    '''same with `RED` but with compressed data, and the data is in signed RGTC1 format.'''
    COMPRESSED_RG_RGTC2 = TextureInternalFormatItem(gl.GL_COMPRESSED_RG_RGTC2, TextureDataType.UNSIGNED_BYTE)
    '''same with `RG` but with compressed data, and the data is in RGTC2 format.'''
    COMPRESSED_SIGNED_RG_RGTC2 = TextureInternalFormatItem(gl.GL_COMPRESSED_SIGNED_RG_RGTC2, TextureDataType.BYTE)
    '''same with `RG` but with compressed data, and the data is in signed RGTC2 format.'''
    COMPRESSED_RGBA_BPTC_UNORM = TextureInternalFormatItem(gl.GL_COMPRESSED_RGBA_BPTC_UNORM, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with compressed data, and the data is in BPTC format, and the data is in unsigned normalized format.'''
    COMPRESSED_SRGB_A_BPTC_UNORM = TextureInternalFormatItem(gl.GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, TextureDataType.UNSIGNED_BYTE)
    '''same with `RGBA` but with compressed data, and the data is in BPTC format, and the data is in sRGB color space, and the data is in unsigned normalized format.'''
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT = TextureInternalFormatItem(gl.GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, TextureDataType.FLOAT)
    '''same with `RGB` but with compressed data, and the data is in signed BPTC format.'''
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = TextureInternalFormatItem(gl.GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, TextureDataType.FLOAT)
    '''same with `RGB` but with compressed data, and the data is in unsigned BPTC format.'''


@dataclass
class TextureFormatType:
    '''dataclass for `TextureFormat` to specify the real GL enum and the default internal format.'''
    gl_format: GL_Constant
    '''OpenGL format enum'''
    PIL_convert_mode: str
    '''PIL image convert mode, e.g. gl.RGBA -> PIL 'RGBA' convert mode.'''
    default_gl_internal_format: TextureInternalFormat
    '''The default internal format when loading texture data to GPU. It specifies the number of color components/color order of the texture data.'''

class TextureFormat(Enum):
    '''Format of texture data. Specifies the number of color components/color order of the texture data.'''

    RED = TextureFormatType(gl.GL_RED, 'L', TextureInternalFormat.RED)
    RED_INT = TextureFormatType(gl.GL_RED_INTEGER, 'L', TextureInternalFormat.RED)
    
    RG = TextureFormatType(gl.GL_RG, 'LA', TextureInternalFormat.RG)
    RG_INT = TextureFormatType(gl.GL_RG_INTEGER, 'LA', TextureInternalFormat.RG)
    
    RGB = TextureFormatType(gl.GL_RGB, 'RGB', TextureInternalFormat.RGB)
    RGB_INT = TextureFormatType(gl.GL_RGB_INTEGER, 'RGB', TextureInternalFormat.RGB)
    
    BGR = TextureFormatType(gl.GL_BGR, 'BGR', TextureInternalFormat.RGB)
    BGR_INT = TextureFormatType(gl.GL_BGR_INTEGER, 'BGR', TextureInternalFormat.RGB)
    
    BGRA = TextureFormatType(gl.GL_BGRA, 'RGBA', TextureInternalFormat.RGBA)
    BGRA_INT = TextureFormatType(gl.GL_BGRA_INTEGER, 'RGBA', TextureInternalFormat.RGBA)
    
    RGBA = TextureFormatType(gl.GL_RGBA, 'RGBA', TextureInternalFormat.RGBA)
    RGBA_INT = TextureFormatType(gl.GL_RGBA_INTEGER, 'RGBA', TextureInternalFormat.RGBA)
    
    DEPTH = TextureFormatType(gl.GL_DEPTH_COMPONENT, 'L', TextureInternalFormat.DEPTH)
    '''for shadow mapping, depth buffer, etc.'''
    DEPTH_STENCIL = TextureFormatType(gl.GL_DEPTH_STENCIL, 'L', TextureInternalFormat.DEPTH_STENCIL)
    '''for depth and stencil buffer, etc.'''
    STENCIL_INDEX = TextureFormatType(gl.GL_STENCIL_INDEX, 'L', TextureInternalFormat.DEPTH_STENCIL)
    '''for stencil buffer, etc.'''

    @property
    def channel_count(self)->int:
        '''Get the number of color channels of this texture format.'''
        return len(self.value.PIL_convert_mode)
    
    @property
    def PIL_convert_mode(self):
        '''Get the PIL image convert mode for this texture format. e.g. gl.RGBA -> PIL 'RGBA' convert mode.'''
        return self.value.PIL_convert_mode
 
    @property
    def default_internal_format(self)->TextureInternalFormat:
        '''
        Get the internal format when loading texture data to GPU.
        `internal format` specifies the number of color components/color order of the texture data,
        while `format` specifies the data type/size of the texture data.
        '''
        return self.value.default_gl_internal_format

    @property
    def default_gl_internal_format(self)->GL_Constant:
        return self.default_internal_format.value.gl_internal_format

    @property
    def default_data_type(self)->TextureDataType:
        return self.default_internal_format.value.default_data_type

    @property
    def default_gl_data_type(self)->GL_Constant:
        return self.default_internal_format.value.default_gl_data_type



__all__.extend(['TextureWrap', 'TextureFilter', 'TextureDataType', 'TextureInternalFormat', 'TextureFormat'])



class PrimitiveDrawingType(Enum):
    """Primitive drawing types of OpenGL"""
    POINTS = gl.GL_POINTS
    LINES = gl.GL_LINES
    LINE_STRIP = gl.GL_LINE_STRIP
    LINE_LOOP = gl.GL_LINE_LOOP
    TRIANGLES = gl.GL_TRIANGLES
    TRIANGLE_STRIP = gl.GL_TRIANGLE_STRIP
    TRIANGLE_FAN = gl.GL_TRIANGLE_FAN
    QUADS = gl.GL_QUADS
    QUAD_STRIP = gl.GL_QUAD_STRIP
    POLYGON = gl.GL_POLYGON

class ShaderType(Enum):
    """Shader types of OpenGL"""
    VERTEX = gl.GL_VERTEX_SHADER
    FRAGMENT = gl.GL_FRAGMENT_SHADER
    GEOMETRY = gl.GL_GEOMETRY_SHADER
    TESS_CONTROL = gl.GL_TESS_CONTROL_SHADER
    TESS_EVALUATION = gl.GL_TESS_EVALUATION_SHADER
    COMPUTE = gl.GL_COMPUTE_SHADER

__all__.extend(['PrimitiveDrawingType', 'ShaderType'])
# endregion