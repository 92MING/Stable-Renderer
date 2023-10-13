from OpenGL.GL import *
from utils.base_clses import NamedObj
from .enums import ShaderType
import glm
from runtime.engineObj import EngineObj

class Shader(NamedObj, EngineObj):

    def __init__(self, name, vertex_source_path:str, fragment_source_path:str):
        super().__init__(name)
        self._vertex_source = open(vertex_source_path, 'r').read()
        self._fragment_source = open(fragment_source_path, 'r').read()
        self._v_shaderID = self._init_shader(self._vertex_source, ShaderType.VERTEX)
        self._f_shaderID = self._init_shader(self._fragment_source, ShaderType.FRAGMENT)
        self._programID = self._init_program(self._v_shaderID, self._f_shaderID)
        print('Loaded shader: ', name)

    def _init_shader(self, source:str, type:ShaderType):
        shaderID = glCreateShader(type.value)
        if shaderID == 0:
            self.engine.RenderManager.printOpenGLError()
            raise RuntimeError(f'Failed to create shader {self.name}')
        glShaderSource(shaderID, source)
        glCompileShader(shaderID)
        # check shader compile status
        if glGetShaderiv(shaderID, GL_COMPILE_STATUS, None) == GL_FALSE:
            info_log = glGetShaderInfoLog(shaderID)
            raise Exception(f'Failed to compile shader {self.name}. Error msg: {info_log}')
        return shaderID
    def _init_program(self, v_shaderID, f_shaderID):
        program = glCreateProgram()
        if program == 0:
            self.engine.RenderManager.printOpenGLError()
            raise RuntimeError(f'Failed to create program when initializing shader: {self.name}')
        glAttachShader(program, v_shaderID)
        glAttachShader(program, f_shaderID)
        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
            raise RuntimeError(f'Failed to link program when initializing shader: {self.name}')

        # try to bind uniform blocks
        matrixBlockIndex = glGetUniformBlockIndex(program, "Matrices")
        if matrixBlockIndex != GL_INVALID_INDEX:
            glUniformBlockBinding(program, matrixBlockIndex, self.engine.RenderManager.MatrixUBO_BindingPoint)

        return program

    # region properties
    @property
    def programID(self):
        return self._programID
    @property
    def vertexSource(self):
        return self._vertex_source
    @property
    def fragmentSource(self):
        return self._fragment_source
    @property
    def vertexShaderID(self):
        return self._v_shaderID
    @property
    def fragmentShaderID(self):
        return self._f_shaderID
    # endregion

    def useProgram(self):
        if glUseProgram(self._programID):
            self.engine.RenderManager.printOpenGLError()

    def getUniformID(self, name:str):
        return glGetUniformLocation(self._programID, name)
    def setUniform(self, name, value):
        self.useProgram()
        val_id = self.getUniformID(name)
        if isinstance(value, (tuple, list)):
            if len(value) == 1:
                value = value[0]
            elif len(value) == 2:
                value = glm.vec2(value)
            elif len(value) == 3:
                value = glm.vec3(value)
            elif len(value) == 4:
                value = glm.vec4(value)
            else:
                raise Exception(f'Invalid uniform value: {value}')
        if isinstance(value, glm.uint32):
            return glUniform1ui(val_id, value)
        if isinstance(value, (int, glm.int32)):
            return glUniform1i(val_id, value)
        elif isinstance(value, float):
            return glUniform1f(val_id, value)
        elif isinstance(value, bool):
            return glUniform1i(val_id, int(value))
        elif isinstance(value, glm.vec1):
            if isinstance(value, glm.dvec1):
                return glUniform1d(val_id, value.x)
            elif isinstance(value, glm.ivec1):
                return glUniform1i(val_id, value.x)
            elif isinstance(value, glm.uvec1):
                return glUniform1ui(val_id, value.x)
            else:
                return glUniform1f(val_id, value.x)
        elif isinstance(value, glm.vec2):
            if isinstance(value, glm.dvec2):
                return glUniform2d(val_id, value.x, value.y)
            elif isinstance(value, glm.ivec2):
                return glUniform2i(val_id, value.x, value.y)
            elif isinstance(value, glm.uvec2):
                return glUniform2ui(val_id, value.x, value.y)
            else:
                return glUniform2f(val_id, value.x, value.y)
        elif isinstance(value, glm.vec3):
            if isinstance(value, glm.dvec3):
                return glUniform3d(val_id, value.x, value.y, value.z)
            elif isinstance(value, glm.ivec3):
                return glUniform3i(val_id, value.x, value.y, value.z)
            elif isinstance(value, glm.uvec3):
                return glUniform3ui(val_id, value.x, value.y, value.z)
            else:
                return glUniform3f(val_id, value.x, value.y, value.z)
        elif isinstance(value, glm.vec4):
            if isinstance(value, glm.dvec4):
                return glUniform4d(val_id, value.x, value.y, value.z, value.w)
            elif isinstance(value, glm.ivec4):
                return glUniform4i(val_id, value.x, value.y, value.z, value.w)
            elif isinstance(value, glm.uvec4):
                return glUniform4ui(val_id, value.x, value.y, value.z, value.w)
            else:
                return glUniform4f(val_id, value.x, value.y, value.z, value.w)
        elif isinstance(value, glm.mat2):
            if isinstance(value, glm.dmat2):
                return glUniformMatrix2dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix2fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3):
            if isinstance(value, glm.dmat3):
                return glUniformMatrix3dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix3fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4):
            if isinstance(value, glm.dmat4):
                return glUniformMatrix4dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix4fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat2x3):
            if isinstance(value, glm.dmat2x3):
                return glUniformMatrix2x3dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix2x3fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat2x4):
            if isinstance(value, glm.dmat2x4):
                return glUniformMatrix2x4dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix2x4fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3x2):
            if isinstance(value, glm.dmat3x2):
                return glUniformMatrix3x2dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix3x2fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3x4):
            if isinstance(value, glm.dmat3x4):
                return glUniformMatrix3x4dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix3x4fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4x2):
            if isinstance(value, glm.dmat4x2):
                return glUniformMatrix4x2dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix4x2fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4x3):
            if isinstance(value, glm.dmat4x3):
                return glUniformMatrix4x3dv(val_id, 1, GL_FALSE, glm.value_ptr(value))
            else:
                return glUniformMatrix4x3fv(val_id, 1, GL_FALSE, glm.value_ptr(value))
        else:
            raise TypeError("Invalid uniform type: {}".format(type(value)))
