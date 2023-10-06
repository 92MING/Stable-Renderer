# -*- coding: utf-8 -*-
'''this module contains enums, constants, etc.'''

from enum import Enum
import OpenGL.GL as gl

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

__all__ = ['PrimitiveType', 'ShaderType']