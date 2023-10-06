from OpenGL.GL import *
from utils.base_clses import NamedObj
from .enums import ShaderType
from static.data_types.vector import Vector

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
    def setUniform(self, name:str, value):
        uni_id = self._getUniformID(name)
        if isinstance(value, Vector):
            if value.dimension == 1:
                ...