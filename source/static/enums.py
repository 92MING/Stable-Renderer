# -*- coding: utf-8 -*-
'''this module contains enums, constants, etc.'''
from enum import Enum
import OpenGL.GL as gl
import OpenGL.GLUT as glut

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

class DepthFunc(Enum):
    NEVER = gl.GL_NEVER
    LESS = gl.GL_LESS
    EQUAL = gl.GL_EQUAL
    LEQUAL = gl.GL_LEQUAL
    GREATER = gl.GL_GREATER
    NOTEQUAL = gl.GL_NOTEQUAL
    GEQUAL = gl.GL_GEQUAL
    ALWAYS = gl.GL_ALWAYS

# region mouse & keyboard
class _FindableEnum(Enum):
    @classmethod
    def GetEnum(cls, value):
        for e in cls:
            if e.value == value:
                return e
        return None
class MouseButton(_FindableEnum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2
    SCROLL_UP = 3
    SCROLL_DOWN = 4
class MouseState(_FindableEnum):
    PRESS = 0
    RELEASE = 1

class Key(_FindableEnum):
    """Key codes of keyboard"""
    SPACE = b' '
    APOSTROPHE = b"'"
    COMMA = b','
    MINUS = b'-'
    PERIOD = b'.'
    SLASH = b'/'
    NUM_0 = b'0'
    NUM_1 = b'1'
    NUM_2 = b'2'
    NUM_3 = b'3'
    NUM_4 = b'4'
    NUM_5 = b'5'
    NUM_6 = b'6'
    NUM_7 = b'7'
    NUM_8 = b'8'
    NUM_9 = b'9'
    SEMICOLON = b';'
    EQUAL = b'='
    A = b'a'
    B = b'b'
    C = b'c'
    D = b'd'
    E = b'e'
    F = b'f'
    G = b'g'
    H = b'h'
    I = b'i'
    J = b'j'
    K = b'k'
    L = b'l'
    M = b'm'
    N = b'n'
    O = b'o'
    P = b'p'
    Q = b'q'
    R = b'r'
    S = b's'
    T = b't'
    U = b'u'
    V = b'v'
    W = b'w'
    X = b'x'
    Y = b'y'
    Z = b'z'
    LEFT_BRACKET = b'['
    BACKSLASH = b'\\'
    RIGHT_BRACKET = b']'
    GRAVE_ACCENT = b'`'
    WORLD_1 = b'world_1'
    WORLD_2 = b'world_2'
    ESCAPE = b'\x1b'
    ENTER = b'\r'
    TAB = b'\t'
    BACKSPACE = b'\x08'
    INSERT = b'insert'
    DELETE = b'delete'
    RIGHT = b'right'
    LEFT = b'left'
    DOWN = b'down'
    UP = b'up'
    PAGE_UP = b'page_up'
    PAGE_DOWN = b'page_down'
    HOME = b'home'
    END = b'end'
    CAPS_LOCK = b'caps_lock'
class SpecialKey(_FindableEnum):
    """Special key codes of keyboard"""
    F1 = glut.GLUT_KEY_F1
    F2 = glut.GLUT_KEY_F2
    F3 = glut.GLUT_KEY_F3
    F4 = glut.GLUT_KEY_F4
    F5 = glut.GLUT_KEY_F5
    F6 = glut.GLUT_KEY_F6
    F7 = glut.GLUT_KEY_F7
    F8 = glut.GLUT_KEY_F8
    F9 = glut.GLUT_KEY_F9
    F10 = glut.GLUT_KEY_F10
    F11 = glut.GLUT_KEY_F11
    F12 = glut.GLUT_KEY_F12
    LEFT = glut.GLUT_KEY_LEFT
    UP = glut.GLUT_KEY_UP
    RIGHT = glut.GLUT_KEY_RIGHT
    DOWN = glut.GLUT_KEY_DOWN
    PAGE_UP = glut.GLUT_KEY_PAGE_UP
    PAGE_DOWN = glut.GLUT_KEY_PAGE_DOWN
    HOME = glut.GLUT_KEY_HOME
    END = glut.GLUT_KEY_END
    INSERT = glut.GLUT_KEY_INSERT
    REPEAT_ON = glut.GLUT_KEY_REPEAT_ON
    REPEAT_OFF = glut.GLUT_KEY_REPEAT_OFF
    REPEAT_DEFAULT = glut.GLUT_KEY_REPEAT_DEFAULT

class KeyState(_FindableEnum):
    PRESS = 0
    RELEASE = 1
    REPEAT = 2
# endregion