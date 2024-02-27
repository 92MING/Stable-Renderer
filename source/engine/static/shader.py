import OpenGL.GL as gl
from utils.base_clses import NamedObj
from .enums import ShaderType
import glm, os
from utils.path_utils import SHADER_DIR
from engine.runtime.engineObj import EngineObj
import re
from typing import Union
from glm import vec1, vec2, vec3, vec4, mat2, mat3, mat4, mat2x3, mat2x4, mat3x2, mat3x4, mat4x2, mat4x3
from utils.type_utils import valueTypeCheck
from typing import get_args

Supported_Shader_Value_Types = Union[int, float, bool, vec1, vec2, vec3, vec4, mat2, mat3, mat4, mat2x3, mat2x4, mat3x2, mat3x4, mat4x2, mat4x3]

class Shader(NamedObj, EngineObj):
    '''
    A class representing a shader program. It contains vertex shader and fragment shader.
    TODO: support other shader types, e.g. geometry shader, tessellation shader.
    '''

    # region class methods & variables
    _Default_GBuffer_Shader = None
    _Default_Defer_Shader = None
    _Default_Post_Shader = None
    _Debug_Shader = None
    _Current_Shader = None
    @staticmethod
    def CurrentShader():
        return Shader._Current_Shader
    # endregion

    def __init__(self, name, vertex_source_path:str, fragment_source_path:str):
        super().__init__(name)
        self._vertex_source = self._edit_shader_source_code(open(vertex_source_path, 'r').read())
        self._fragment_source = self._edit_shader_source_code(open(fragment_source_path, 'r').read())
        self._v_shaderID = self._init_shader(self._vertex_source, ShaderType.VERTEX)
        self._f_shaderID = self._init_shader(self._fragment_source, ShaderType.FRAGMENT)
        self._programID = self._init_program(self._v_shaderID, self._f_shaderID)
        print('Loaded shader: ', name)

    def _edit_shader_constant(self, source:str, constant_name:str, value):
        pattern = r'#[ ]*?define[ ]+?'+constant_name+r'[ ]+?(\d+)'
        if len(re.findall(pattern, source)) > 0:
            source_code = re.sub(pattern, f'#define {constant_name} {value}', source)
        else:
            glslLines = source.split('\n')
            find_version_pattern = r'#[ ]*?version[ ]+?(\d+)[ ]+?(core|compatibility|es|glsl)?'
            for i in range(len(glslLines)):
                if len(re.findall(find_version_pattern, glslLines[i])) > 0:
                    glslLines.insert(i + 1, f'#define {constant_name} {value}')
                    break
                if i == len(glslLines) - 1:
                    glslLines.insert(0, f'#define {constant_name} {value}')
            source_code = '\n'.join(glslLines)
        return source_code

    def _edit_shader_source_code(self, source_code:str):
        '''edit shader source code's constant value,...'''
        from ..runtime.components.light.light import Light
        for subLightType in Light.AllLightSubTypes():
            # edit max light number constant. E.g. #define MAX_POINTLIGHT 256
            name = subLightType.__qualname__.split('.')[-1].upper()
            self._edit_shader_constant(source_code, f'MAX_{name}', subLightType.Max_Num())
        return source_code

    def _init_shader(self, source:str, type:ShaderType):
        shaderID = gl.glCreateShader(type.value)
        if shaderID == 0:
            self.engine.PrintOpenGLError()
            raise RuntimeError(f'Failed to create shader {self.name}')
        gl.glShaderSource(shaderID, source)
        gl.glCompileShader(shaderID)
        # check shader compile status
        if gl.glGetShaderiv(shaderID, gl.GL_COMPILE_STATUS, None) == gl.GL_FALSE:
            info_log = gl.glGetShaderInfoLog(shaderID)
            raise Exception(f'Failed to compile shader {self.name}. Error msg: {info_log}')
        return shaderID

    def _init_program(self, v_shaderID, f_shaderID):
        program = gl.glCreateProgram()
        if program == 0:
            self.engine.PrintOpenGLError()
            raise RuntimeError(f'Failed to create program when initializing shader: {self.name}')
        gl.glAttachShader(program, v_shaderID)
        gl.glAttachShader(program, f_shaderID)
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS, None) == gl.GL_FALSE:
            raise RuntimeError(f'Failed to link program when initializing shader: {self.name}')

        # try to bind uniform blocks
        engineBlockIndex = gl.glGetUniformBlockIndex(program, "Engine") # engine block contains engine settings, e.g. screen size
        if engineBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, engineBlockIndex, self.engine.RuntimeManager.EngineUBO_BindingPoint)
        matrixBlockIndex = gl.glGetUniformBlockIndex(program, "Matrices")
        if matrixBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, matrixBlockIndex, self.engine.RuntimeManager.MatrixUBO_BindingPoint)
        lightBlockIndex = gl.glGetUniformBlockIndex(program, "Lights")
        if lightBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, lightBlockIndex, self.engine.RuntimeManager.LightUBO_BindingPoint)

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
        if self.CurrentShader() == self:
            return
        if gl.glUseProgram(self._programID):
            self.engine.PrintOpenGLError()
            raise RuntimeError(f'Failed to use shader {self.name}')
        Shader._Current_Shader = self

    def getUniformID(self, name:str):
        return gl.glGetUniformLocation(self._programID, name)
    def setUniform(self, name, value:Supported_Shader_Value_Types):
        if not valueTypeCheck(value, Supported_Shader_Value_Types):
            raise TypeError(f'Invalid uniform value type: {type(value)}. Supported types: {get_args(Supported_Shader_Value_Types)}')
        self.useProgram()
        val_id = self.getUniformID(name)
        if val_id == -1:
            return
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
            return gl.glUniform1ui(val_id, value)
        if isinstance(value, (int, glm.int32)):
            return gl.glUniform1i(val_id, value)
        elif isinstance(value, float):
            return gl.glUniform1f(val_id, value)
        elif isinstance(value, bool):
            return gl.glUniform1i(val_id, int(value))
        elif isinstance(value, glm.vec1):
            if isinstance(value, glm.dvec1):
                return gl.glUniform1d(val_id, value.x)
            elif isinstance(value, glm.ivec1):
                return gl.glUniform1i(val_id, value.x)
            elif isinstance(value, glm.uvec1):
                return gl.glUniform1ui(val_id, value.x)
            else:
                return gl.glUniform1f(val_id, value.x)
        elif isinstance(value, glm.vec2):
            if isinstance(value, glm.dvec2):
                return gl.glUniform2d(val_id, value.x, value.y)
            elif isinstance(value, glm.ivec2):
                return gl.glUniform2i(val_id, value.x, value.y)
            elif isinstance(value, glm.uvec2):
                return gl.glUniform2ui(val_id, value.x, value.y)
            else:
                return gl.glUniform2f(val_id, value.x, value.y)
        elif isinstance(value, glm.vec3):
            if isinstance(value, glm.dvec3):
                return gl.glUniform3d(val_id, value.x, value.y, value.z)
            elif isinstance(value, glm.ivec3):
                return gl.glUniform3i(val_id, value.x, value.y, value.z)
            elif isinstance(value, glm.uvec3):
                return gl.glUniform3ui(val_id, value.x, value.y, value.z)
            else:
                return gl.glUniform3f(val_id, value.x, value.y, value.z)
        elif isinstance(value, glm.vec4):
            if isinstance(value, glm.dvec4):
                return gl.glUniform4d(val_id, value.x, value.y, value.z, value.w)
            elif isinstance(value, glm.ivec4):
                return gl.glUniform4i(val_id, value.x, value.y, value.z, value.w)
            elif isinstance(value, glm.uvec4):
                return gl.glUniform4ui(val_id, value.x, value.y, value.z, value.w)
            else:
                return gl.glUniform4f(val_id, value.x, value.y, value.z, value.w)
        elif isinstance(value, glm.mat2):
            if isinstance(value, glm.dmat2):
                return gl.glUniformMatrix2dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix2fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3):
            if isinstance(value, glm.dmat3):
                return gl.glUniformMatrix3dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix3fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4):
            if isinstance(value, glm.dmat4):
                return gl.glUniformMatrix4dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix4fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat2x3):
            if isinstance(value, glm.dmat2x3):
                return gl.glUniformMatrix2x3dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix2x3fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat2x4):
            if isinstance(value, glm.dmat2x4):
                return gl.glUniformMatrix2x4dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix2x4fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3x2):
            if isinstance(value, glm.dmat3x2):
                return gl.glUniformMatrix3x2dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix3x2fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat3x4):
            if isinstance(value, glm.dmat3x4):
                return gl.glUniformMatrix3x4dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix3x4fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4x2):
            if isinstance(value, glm.dmat4x2):
                return gl.glUniformMatrix4x2dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix4x2fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        elif isinstance(value, glm.mat4x3):
            if isinstance(value, glm.dmat4x3):
                return gl.glUniformMatrix4x3dv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
            else:
                return gl.glUniformMatrix4x3fv(val_id, 1, gl.GL_FALSE, glm.value_ptr(value))
        else:
            raise TypeError("Invalid uniform type: {}".format(type(value)))

    # region default shaders
    @classmethod
    def Default_GBuffer_Shader(cls):
        if cls._Default_GBuffer_Shader is None:
            cls._Default_GBuffer_Shader = Shader("default_Gbuffer_shader",
                                                  os.path.join(SHADER_DIR, "default_Gbuffer.vert.glsl"),
                                                  os.path.join(SHADER_DIR, "default_Gbuffer.frag.glsl"))
        return cls._Default_GBuffer_Shader
    @classmethod
    def Default_Defer_Shader(cls):
        if cls._Default_Defer_Shader is None:
            cls._Default_Defer_Shader = Shader("default_defer_render_shader",
                                                os.path.join(SHADER_DIR, "default_defer_render.vert.glsl"),
                                                os.path.join(SHADER_DIR, "default_defer_render.frag.glsl"))
        return cls._Default_Defer_Shader
    @classmethod
    def Default_Post_Shader(cls):
        if cls._Default_Post_Shader is None:
            cls._Default_Post_Shader = Shader("default_post_process",
                                              os.path.join(SHADER_DIR, 'default_post_process.vert.glsl'),
                                              os.path.join(SHADER_DIR, 'default_post_process.frag.glsl'))
        return cls._Default_Post_Shader
    @classmethod
    def Debug_Shader(cls):
        if cls._Debug_Shader is None:
            cls._Debug_Shader = Shader("debug_shader",
                                        os.path.join(SHADER_DIR, "debug.vert.glsl"),
                                        os.path.join(SHADER_DIR, "debug.frag.glsl"))
        return cls._Debug_Shader
    # endregion


__all__ = ['Shader','Supported_Shader_Value_Types']