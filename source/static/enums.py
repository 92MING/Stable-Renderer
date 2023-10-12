# -*- coding: utf-8 -*-
'''this module contains enums, constants, etc.'''
from enum import Enum
import OpenGL.GL as gl

# region opengl
class DepthFunc(Enum):
    NEVER = gl.GL_NEVER
    LESS = gl.GL_LESS
    EQUAL = gl.GL_EQUAL
    LEQUAL = gl.GL_LEQUAL
    GREATER = gl.GL_GREATER
    NOTEQUAL = gl.GL_NOTEQUAL
    GEQUAL = gl.GL_GEQUAL
    ALWAYS = gl.GL_ALWAYS
# endregion

# region render
class RenderOrder:
    """
    Render order of GameObjects.
    Note that this is not a enum class.
    """
    OPAQUE = 1000
    TRANSPARENT = 2000
    OVERLAY = 3000

class ProjectionType(Enum):
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1

class LightType(Enum):
    DIRECTIONAL_LIGHT = 0
    POINT_LIGHT = 1
    SPOT_LIGHT = 2
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
    NONE = 0
    SHIFT = 1
    CTRL = 2
    ALT = 4
    SUPER = 8
class Key(_FindableEnum):
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
# endregion

# region resource
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
class TextureFormat(Enum):
    RED = gl.GL_RED
    RG = gl.GL_RG
    RGB = gl.GL_RGB
    BGR = gl.GL_BGR
    RGBA = gl.GL_RGBA
    BGRA = gl.GL_BGRA
    DEPTH_COMPONENT = gl.GL_DEPTH_COMPONENT
    DEPTH_STENCIL = gl.GL_DEPTH_STENCIL
class TextureType(Enum):
    UNSIGNED_BYTE = gl.GL_UNSIGNED_BYTE
    BYTE = gl.GL_BYTE
    UNSIGNED_SHORT = gl.GL_UNSIGNED_SHORT
    SHORT = gl.GL_SHORT
    UNSIGNED_INT = gl.GL_UNSIGNED_INT
    INT = gl.GL_INT
    FLOAT = gl.GL_FLOAT
    UNSIGNED_BYTE_3_3_2 = gl.GL_UNSIGNED_BYTE_3_3_2
    UNSIGNED_BYTE_2_3_3_REV = gl.GL_UNSIGNED_BYTE_2_3_3_REV
    UNSIGNED_SHORT_5_6_5 = gl.GL_UNSIGNED_SHORT_5_6_5
    UNSIGNED_SHORT_5_6_5_REV = gl.GL_UNSIGNED_SHORT_5_6_5_REV
    UNSIGNED_SHORT_4_4_4_4 = gl.GL_UNSIGNED_SHORT_4_4_4_4
    UNSIGNED_SHORT_4_4_4_4_REV = gl.GL_UNSIGNED_SHORT_4_4_4_4_REV
    UNSIGNED_SHORT_5_5_5_1 = gl.GL_UNSIGNED_SHORT_5_5_5_1
    UNSIGNED_SHORT_1_5_5_5_REV = gl.GL_UNSIGNED_SHORT_1_5_5_5_REV
    UNSIGNED_INT_8_8_8_8 = gl.GL_UNSIGNED_INT_8_8_8_8
    UNSIGNED_INT_8_8_8_8_REV = gl.GL_UNSIGNED_INT_8_8_8_8_REV
    UNSIGNED_INT_10_10_10_2 = gl.GL_UNSIGNED_INT_10_10_10_2
    UNSIGNED_INT_2_10_10_10_REV = gl.GL_UNSIGNED_INT_2_10_10_10_REV

class PrimitiveType(Enum):
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
# endregion