from OpenGL.GL import *
from utils.base_clses import NamedObj
from .enums import ShaderType
from static.data_types.vector import Vector
from static.data_types.matrix import Matrix
from .data_types.base_types import *

class Shader(NamedObj):

    def __init__(self, name, vertex_source_path:str, fragment_source_path:str):
        super().__init__(name)
        self._vertex_source = open(vertex_source_path, 'r').read()
        self._fragment_source = open(fragment_source_path, 'r').read()
        self._v_shaderID = self._init_shader(self._vertex_source, ShaderType.VERTEX)
        self._f_shaderID = self._init_shader(self._fragment_source, ShaderType.FRAGMENT)
        self._programID = self._init_program(self._v_shaderID, self._f_shaderID)

    def _init_shader(self, source:str, type:ShaderType):
        shaderID = glCreateShader(type.value)
        if shaderID == 0:
            raise RuntimeError(f'Failed to create shader {self.name}')
        glShaderSource(shaderID, source)
        glCompileShader(shaderID)
        # check shader compile status
        if glGetShaderiv(shaderID, GL_COMPILE_STATUS, None) == GL_FALSE:
            info_log = glGetShaderInfoLog(shaderID)
            glDeleteProgram(shaderID)
            raise Exception(f'Failed to compile shader {self.name}. Error msg: {info_log}')
        return shaderID
    def _init_program(self, v_shaderID, f_shaderID):
        program = glCreateProgram()
        if program == 0:
            raise RuntimeError(f'Failed to create program when initializing shader: {self.name}')
        glAttachShader(program, v_shaderID)
        glAttachShader(program, f_shaderID)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
            glDeleteProgram(program)
            raise RuntimeError(f'Failed to link program when initializing shader: {self.name}')
        return program

    def useProgram(self):
        glUseProgram(self._programID)

    def _getUniformID(self, name:str):
        return glGetUniformLocation(self._programID, name)
    def setUniform(self, name, value):
        val_id = self._getUniformID(name)
        if isinstance(value, int):
            return glUniform1i(val_id, value)
        elif isinstance(value, float):
            return glUniform1f(val_id, value)
        elif isinstance(value, bool):
            return glUniform1i(val_id, int(value))
        elif isinstance(value, Uint):
            return glUniform1ui(val_id, value)
        elif isinstance(value, Double):
            return glUniform1d(val_id, value)
        elif isinstance(value, Vector):
            if value.dimension == 2:
                if value.dtype == Int:
                    return glUniform2i(val_id, value.x, value.y)
                elif value.dtype == Float:
                    return glUniform2f(val_id, value.x, value.y)
                elif value.dtype == Uint:
                    return glUniform2ui(val_id, value.x, value.y)
                elif value.dtype == Double:
                    return glUniform2d(val_id, value.x, value.y)
            elif value.dimension == 3:
                if value.dtype == Int:
                    return glUniform3i(val_id, value.x, value.y, value.z)
                elif value.dtype == Float:
                    return glUniform3f(val_id, value.x, value.y, value.z)
                elif value.dtype == Uint:
                    return glUniform3ui(val_id, value.x, value.y, value.z)
                elif value.dtype == Double:
                    return glUniform3d(val_id, value.x, value.y, value.z)
            elif value.dimension == 4:
                if value.dtype == Int:
                    return glUniform4i(val_id, value.x, value.y, value.z, value.w)
                elif value.dtype == Float:
                    return glUniform4f(val_id, value.x, value.y, value.z, value.w)
                elif value.dtype == Uint:
                    return glUniform4ui(val_id, value.x, value.y, value.z, value.w)
                elif value.dtype == Double:
                    return glUniform4d(val_id, value.x, value.y, value.z, value.w)
        elif isinstance(value, Matrix):
            if value.shape == (2, 2):
                if value.dtype == Float:
                    return glUniformMatrix2fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix2dv(val_id, 1, GL_FALSE, value.data)
            elif value.shape == (3, 3):
                if value.dtype == Float:
                    return glUniformMatrix3fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix3dv(val_id, 1, GL_FALSE, value.data)
            elif value.shape == (2, 3):
                if value.dtype == Float:
                    return glUniformMatrix2x3fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix2x3dv(val_id, 1, GL_FALSE, value.data)
            elif value.shape == (3, 2):
                if value.dtype == Float:
                    return glUniformMatrix3x2fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix3x2dv(val_id, 1, GL_FALSE, value.data)
            if value.shape == (4, 4):
                if value.dtype == Float:
                    return glUniformMatrix4fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix4dv(val_id, 1, GL_FALSE, value.data)
            if value.shape == (2, 4):
                if value.dtype == Float:
                    return glUniformMatrix2x4fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix2x4dv(val_id, 1, GL_FALSE, value.data)
            if value.shape == (4, 2):
                if value.dtype == Float:
                    return glUniformMatrix4x2fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix4x2dv(val_id, 1, GL_FALSE, value.data)
            if value.shape == (3, 4):
                if value.dtype == Float:
                    return glUniformMatrix3x4fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix3x4dv(val_id, 1, GL_FALSE, value.data)
            if value.shape == (4, 3):
                if value.dtype == Float:
                    return glUniformMatrix4x3fv(val_id, 1, GL_FALSE, value.data)
                elif value.dtype == Double:
                    return glUniformMatrix4x3dv(val_id, 1, GL_FALSE, value.data)
        if not hasattr(value, 'dtype'):
            raise RuntimeError(f'Unsupported uniform type: {type(value)}')
        else:
            raise RuntimeError(f'Unsupported uniform type: {type(value)} with dtype: {value.dtype}')
