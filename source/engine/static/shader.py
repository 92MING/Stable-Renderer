import re
import glm
import os
import OpenGL.GL as gl
from OpenGL.GL import Constant as GL_Const

from dataclasses import dataclass
from io import IOBase
from common_utils.base_clses import NamedObj
from engine.runtime.engineObj import EngineObj
from glm import vec1, vec2, vec3, vec4, mat2, mat3, mat4, mat2x3, mat2x4, mat3x2, mat3x4, mat4x2, mat4x3
from typing import (get_args, Optional, Union, Dict, Any, ClassVar, TypeAlias, TypeVar, List, Type, Generic, ValuesView, 
                    get_origin, Annotated, Sequence, Tuple, TYPE_CHECKING)

from common_utils.path_utils import SHADER_DIR
from common_utils.decorators import class_property
from common_utils.debug_utils import EngineLogger
from common_utils.type_utils import valueTypeCheck
from .enums import ShaderType

if TYPE_CHECKING:
    from engine.managers import RuntimeManager


SupportedShaderValueTypes: TypeAlias = Union[int, float, bool,  
                                             vec1, glm.dvec1, glm.ivec1, glm.uvec1, 
                                             vec2, glm.dvec2, glm.ivec2, glm.uvec2,
                                             vec3, glm.dvec3, glm.ivec3, glm.uvec3,
                                             vec4, glm.dvec4, glm.ivec4, glm.uvec4,
                                             mat2, glm.dmat2, 
                                             mat3, glm.dmat3,
                                             mat4, glm.dmat4,
                                             mat2x3, glm.dmat2x3,
                                             mat2x4, glm.dmat2x4,
                                             mat3x2, glm.dmat3x2,
                                             mat3x4, glm.dmat3x4,
                                             mat4x2, glm.dmat4x2,
                                             mat4x3, glm.dmat4x3,
                                             'ShaderStruct']
'''Supported types of shader uniform values'''

_ShaderTypeSizeMapping = {
    int: 4, float: 4, bool: 4,
    vec1: glm.sizeof(glm.vec1), glm.dvec1: glm.sizeof(glm.dvec1), glm.ivec1: glm.sizeof(glm.ivec1), glm.uvec1: glm.sizeof(glm.uvec1),
    vec2: glm.sizeof(glm.vec2), glm.dvec2: glm.sizeof(glm.dvec2), glm.ivec2: glm.sizeof(glm.ivec2), glm.uvec2: glm.sizeof(glm.uvec2),
    vec3: glm.sizeof(glm.vec3), glm.dvec3: glm.sizeof(glm.dvec3), glm.ivec3: glm.sizeof(glm.ivec3), glm.uvec3: glm.sizeof(glm.uvec3),
    vec4: glm.sizeof(glm.vec4), glm.dvec4: glm.sizeof(glm.dvec4), glm.ivec4: glm.sizeof(glm.ivec4), glm.uvec4: glm.sizeof(glm.uvec4),
    mat2: glm.sizeof(glm.mat2), glm.dmat2: glm.sizeof(glm.dmat2),
    mat3: glm.sizeof(glm.mat3), glm.dmat3: glm.sizeof(glm.dmat3),
    mat4: glm.sizeof(glm.mat4), glm.dmat4: glm.sizeof(glm.dmat4),
    mat2x3: glm.sizeof(glm.mat2x3), glm.dmat2x3: glm.sizeof(glm.dmat2x3),
    mat2x4: glm.sizeof(glm.mat2x4), glm.dmat2x4: glm.sizeof(glm.dmat2x4),
    mat3x2: glm.sizeof(glm.mat3x2), glm.dmat3x2: glm.sizeof(glm.dmat3x2),
    mat3x4: glm.sizeof(glm.mat3x4), glm.dmat3x4: glm.sizeof(glm.dmat3x4),
    mat4x2: glm.sizeof(glm.mat4x2), glm.dmat4x2: glm.sizeof(glm.dmat4x2),
    mat4x3: glm.sizeof(glm.mat4x3), glm.dmat4x3: glm.sizeof(glm.dmat4x3),
}
'''The size of each type in shader'''

_ShaderConstants: Dict[str, Union[int, float]] = {}
'''Constants that will be used inner .glsl, e.g. #define MAX_POINTLIGHT 256'''

ST = TypeVar('ST', bound=SupportedShaderValueTypes)

# region shader obj
class ShaderField(Generic[ST]):
    '''A field in shader'''
    
    name: str
    '''The name of the field'''
    type: Type[ST]
    '''The type of the field'''
    default: Union[ST, None] = None
    '''The default value of the field'''
    _value: Union[ST, None] = None
    '''The value of the field'''
    offset: int = 0
    '''The address offset of the field(if it is in a struct)'''
    
    def __init__(self, 
                 name: str, 
                 type: Type[ST], 
                 default: Union[ST, None] = None,
                 offset: int = 0,):
        if not isinstance(type, ShaderStruct):
            if type not in _ShaderTypeSizeMapping:
                raise ValueError(f'Invalid field type: {type}. Should be one of {get_args(SupportedShaderValueTypes)}')
        self.name = name
        self.type = type    # type: ignore
        self.default = default
        self._value = default
        self.offset = offset
    
    @property
    def value(self)->Union[ST, None]:
        return self._value
    
    @value.setter
    def value(self, val: Optional[ST]):
        if val is not None:
            if not valueTypeCheck(val, self.type):  # type: ignore
                raise TypeError(f'Invalid value type: {type(val)}. Should be one of {get_args(self.type)}')
        self._value = val
    
    @property
    def reset(self):
        '''
        Reset the value to the default value.
        This is only available when the `default` value is not None.
        '''
        if self.default is not None:
            self._value = self.default
    
    @property
    def size(self)->int:
        if self.type in _ShaderTypeSizeMapping:
            return _ShaderTypeSizeMapping[self.type]
        return self.type.Size   # type: ignore

    def update(self, 
               shader: 'Shader', 
               name_prefix: str = '',):
        '''
        Update the value in shader program.
        If the field is in a struct, you should pass the name_prefix. e.g. "light.
        
        Args:
            * shader: the shader program
            * name_prefix: the name prefix of the field
        "'''
        if self.value is not None:
            shader.setUniform(name_prefix + self.name, self.value)

class ShaderArrayField(ShaderField[ST]):
    '''A field in shader that is an array'''
    
    max_size: int
    '''the max size of the array'''
    default: Union[List[Union[ST, None]], None]
    '''
    Default value of the array can be a list of values or a single value.
    If it is:
        - a single value: will be broadcasted to the max size.
        - a list of values: the length should be less than or equal to the max size.
    '''
    _value: List[ST]
    '''The value of the array'''
    
    def __init__(self, 
                 name: str, 
                 type: Type[ST], 
                 max_size: int, 
                 default: Union[List[ST], ST, None] = None,
                 offset: int = 0,
                 ):
        if default is not None:
            if isinstance(default, (list, tuple)):
                if len(default) > max_size:
                    raise ValueError(f'Invalid default value: {default}. The length should be less than or equal to the max size: {max_size}')
                for val in default:
                    if not valueTypeCheck(val, type): # type: ignore
                        raise TypeError(f'Invalid default value: {val}. Should be one of {get_args(type)}')
            else:
                if not valueTypeCheck(default, type):
                    raise TypeError(f'Invalid default value: {default}. Should be one of {get_args(type)}')
                default = [default] * max_size
        super().__init__(name=name, type=type, default=default, offset=offset)
        if len(self.value) < max_size:
            self.value = self.value + [None] * (max_size - len(self.value))
        self.max_size = max_size
    
    @property
    def size(self)->int:
        '''the total size of this array field'''
        return super().size * self.max_size
    
    @property
    def value(self)->List[ST]:
        return self._value
    
    @value.setter
    def value(self, val: Sequence[ST]):
        if not isinstance(val, list):
            val = list(val)
        if len(val) > self.max_size:
            raise ValueError(f'Invalid value: {val}. The length should be less than or equal to the max size: {self.max_size}')
        for v in val:
            if v is not None:
                if not valueTypeCheck(v, self.type):
                    raise TypeError(f'Invalid value: {v}. Should be one of {get_args(self.type)}')
        if len(val) < self.max_size:
            val = val + [None] * (self.max_size - len(val))
        self._value = val
    
    def set_at(self, index: int, val: Optional[ST]):
        if index >= self.max_size:
            raise ValueError(f'Invalid index: {index}. Should be less than {self.max_size}')
        if val is not None:
            if not valueTypeCheck(val, self.type):
                raise TypeError(f'Invalid value: {val}. Should be one of {get_args(self.type)}')
        self._value[index] = val
    
    def get_at(self, index: int)->Optional[ST]:
        if index >= self.max_size:
            raise ValueError(f'Invalid index: {index}. Should be less than {self.max_size}')
        return self._value[index]
    
    def __getitem__(self, index: int)->Optional[ST]:
        return self.get_at(index)
    
    def __setitem__(self, index: int, val: Optional[ST]):
        self.set_at(index, val)

    def update_at(self, 
                  shader: 'Shader', 
                  index: int, 
                  name_prefix: str = '',
                  ):
        '''
        Update the value at the specific index in shader program.
        
        Args:
            * shader: the shader program
            * index: the index of the value
        '''
        if index >= self.max_size:
            raise ValueError(f'Invalid index: {index}. Should be less than {self.max_size}')
        name = f'{self.name}[{index}]'
        if self.value[index] is not None:
            shader.setUniform(name_prefix + name, self.value[index])

# TODO: 2d/3d array field

@dataclass
class _ArrayAnnotation:
    type: Type[SupportedShaderValueTypes]
    size: int
    
def Array(type: Type[ST], size: int) -> Annotated:
    '''annotation for array field in shader struct. e.g. `position: Array[vec3, 4]` means a vec3 array with size 4.'''
    if not type in _ShaderTypeSizeMapping and not issubclass(type, ShaderStruct):
        raise ValueError(f'Invalid field type: {type}. Should be one of {get_args(SupportedShaderValueTypes)}')
    return Annotated[type, _ArrayAnnotation(type, size)]

# TODO: 2d/3d array annotation
# endregion

class ShaderStruct:
    '''
    A mapping class for shader struct. It is used to define the structure of a shader struct.
    '''
    
    _Fields: Dict[str, ShaderField]
    '''A dict containing all fields' info. The key is the field name, and the value is the field info.'''
    
    @class_property 
    def Fields(cls)->ValuesView[ShaderField]:
        return cls._Fields.values()
    
    @class_property
    def Size(cls):
        '''return the total size of this struct in bytes'''
        return sum(field.size for field in cls._Fields.values())
    
    def __init_subclass__(cls) -> None:
        cls._Fields = {}
        offset = 0
        for field_name, field_type in cls.__annotations__.items():
            if field_name.startswith('_') or field_name == 'gl_id': # ignore private fields and gl_id
                continue
            
            if hasattr(cls, field_name):
                default = getattr(cls, field_name)
            else:
                default = None
            
            origin = get_origin(field_type)
            if origin:
                if origin == Annotated:
                    field_type_info = field_type.__metadata__[0]
                    if not isinstance(field_type_info, _ArrayAnnotation):
                        raise ValueError(f'Invalid annotation: {field_type_info}. Should be one of {get_args(SupportedShaderValueTypes)}')
                    else:
                        field = ShaderArrayField(name=field_name, 
                                                 inner_type=field_type_info.type, 
                                                 max_size=field_type_info.size, 
                                                 default=default, 
                                                 offset=offset)
                else:
                    raise ValueError(f'Invalid field type: {field_type}. Should be one of {get_args(SupportedShaderValueTypes)}')
            else:
                if field_type not in SupportedShaderValueTypes and not issubclass(field_type, ShaderStruct):
                    raise ValueError(f'Invalid field type: {field_type}. Should be one of {get_args(SupportedShaderValueTypes)}')
                else:
                    field = ShaderField(name=field_name, type=field_type, default=default, offset=offset)
            
            cls._Fields[field_name] = field
            offset += field.size
    
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    def __setattr__(self, name: str, val):
        if name not in self._Fields:
            raise AttributeError(f'Invalid attribute: {name}')
        super().__setattr__(name, val)
    
    def update(self, shader: 'Shader', name: str = ''):
        '''
        Update the value in shader program.
        If the struct is in another struct, you should pass the name_prefix. e.g. "light.
        
        Args:
            * shader: the shader program
            * name: the name of the struct in shader program
        '''
        for field in self._Fields.values():
            field.update(shader=shader, name_prefix=name) 
    
class ShaderBufferStruct(ShaderStruct):
    
    _GL_BUFFER_TYPE: ClassVar[GL_Const]
    '''The OpenGL buffer type, e.g. gl.GL_UNIFORM_BUFFER, gl.GL_SHADER_STORAGE_BUFFER, etc.'''
    _BindingPoint: ClassVar[int]
    '''The binding point of the buffer'''
    
    _in_context: bool = False
    '''
    Whether the buffer is in a context. If it is in a context, it will update until the context is closed.
    You can enter a context by using the `with` statement.
    '''
    gl_id: int
    '''The OpenGL buffer id. Will be generated by glGenBuffers(1) if it is not set in __init__.'''
    
    def __init_subclass__(cls, GLBufferType: GL_Const, BindingPoint: int) -> None:
        cls._GL_BUFFER_TYPE = GLBufferType
        cls._BindingPoint = BindingPoint
        super().__init_subclass__()
    
    def __init__(self, **kwargs):
        gl_id = kwargs.get('gl_id', None)
        if gl_id is None:
            gl_id = gl.glGenBuffers(1)
        self.gl_id = gl_id
        
        self.bind()
        gl.glBufferData(self._GL_BUFFER_TYPE, self.Size, None, gl.GL_DYNAMIC_DRAW)
        
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.update()
        
    def __setattr__(self, name: str, val):
        if name not in self._Fields:
            raise AttributeError(f'Invalid attribute: {name}')
        super().__setattr__(name, val)
        if not self._in_context:    # if in context, we will update at the end of the context
            self.update()

    def __enter__(self):
        self.bind()
        self._in_context = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        self.update()

    def update(self):
        '''
        Update the value in shader buffer.
        Override ShaderStruct.update
        '''
        raise NotImplementedError
        self.bind()
        gl.glBindBufferBase(self._GL_BUFFER_TYPE, self._BindingPoint, self.gl_id)

        for _, field in self._Fields.items():
            gl.glBufferSubData(self._GL_BUFFER_TYPE, 
                               field.offset, 
                               field.size, 
                               None)    # TODO: turn value into pointer
    
    def bind(self):
        gl.glBindBuffer(self._GL_BUFFER_TYPE, self.gl_id)
# endregion


class Shader(NamedObj, EngineObj):
    '''
    A class representing a shader program. It contains vertex shader and fragment shader.
    TODO: support other shader types, e.g. geometry shader, tessellation shader.
    '''

    # region class methods & variables
    _Default_GBuffer_Shader: 'Shader'
    _Default_Defer_Shader: 'Shader'
    _Default_Post_Shader: 'Shader'
    _Debug_Shader: 'Shader'
    _Current_Shader: Optional['Shader'] = None
    '''The last shader that has been used(by calling `useProgram()`). It is used to avoid redundant shader switch.'''
    
    @classmethod
    def Default_GBuffer_Shader(cls):
        if not hasattr(cls, '_Default_GBuffer_Shader'):
            cls._Default_GBuffer_Shader = Shader("default_Gbuffer_shader",
                                                  os.path.join(SHADER_DIR, "default_Gbuffer.vert.glsl"),
                                                  os.path.join(SHADER_DIR, "default_Gbuffer.frag.glsl"))
        return cls._Default_GBuffer_Shader
    
    @classmethod
    def Default_Defer_Shader(cls):
        if not hasattr(cls, '_Default_Defer_Shader'):
            cls._Default_Defer_Shader = Shader("default_defer_render_shader",
                                                os.path.join(SHADER_DIR, "default_defer_render.vert.glsl"),
                                                os.path.join(SHADER_DIR, "default_defer_render.frag.glsl"))
        return cls._Default_Defer_Shader
    
    @classmethod
    def Default_Post_Shader(cls):
        if not hasattr(cls, '_Default_Post_Shader'):
            cls._Default_Post_Shader = Shader("default_post_process",
                                              os.path.join(SHADER_DIR, 'default_post_process.vert.glsl'),
                                              os.path.join(SHADER_DIR, 'default_post_process.frag.glsl'))
        return cls._Default_Post_Shader
    
    @classmethod
    def Debug_Shader(cls):
        if not hasattr(cls, '_Debug_Shader'):
            cls._Debug_Shader = Shader("debug_shader",
                                        os.path.join(SHADER_DIR, "debug.vert.glsl"),
                                        os.path.join(SHADER_DIR, "debug.frag.glsl"))
        return cls._Debug_Shader
    
    @staticmethod
    def CurrentShader()->Optional['Shader']:
        return Shader._Current_Shader
    
    @staticmethod
    def SetShaderConstant(name:str, s):
        '''Set constants that will be used inner .glsl, e.g. #define MAX_POINTLIGHT 256'''
        _ShaderConstants[name] = s
    # endregion

    def __init__(self, 
                 name: str, 
                 vertex_source_path: str, 
                 fragment_source_path: str):
        super().__init__(name)
        with open(vertex_source_path, 'r') as f:
            self._vertex_source = self._edit_shader_source_code(f)
        '''source code of vertex shader'''
        with open(fragment_source_path, 'r') as f:
            self._fragment_source = self._edit_shader_source_code(f)
        '''source code of fragment shader'''
        
        self._v_shaderID = self._init_shader(self._vertex_source, ShaderType.VERTEX)
        self._f_shaderID = self._init_shader(self._fragment_source, ShaderType.FRAGMENT)
        
        self._programID = self._init_program(self._v_shaderID, self._f_shaderID)
        EngineLogger.print('Loaded shader: ', name)
        
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

    def _edit_shader_source_code(self, source_code:Union[str, IOBase]):
        '''edit shader source code, e.g. adding constant values,...'''
        str_source_code: str = source_code.read() if isinstance(source_code, IOBase) else source_code
        for const_name, const_value in _ShaderConstants.items():
            str_source_code = self._edit_shader_constant(str_source_code, const_name, const_value)
        return str_source_code

    def _init_shader(self, source:str, type:ShaderType):
        if type not in (ShaderType.VERTEX, ShaderType.FRAGMENT):
            raise ValueError(f'Invalid shader type: {type}, currently only support vertex and fragment shader')
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
            raise RuntimeError(f'Failed to link program when initializing shader: {self.name}, reason: {gl.glGetProgramInfoLog(program)}')

        # try to bind uniform blocks
        runtimeManager: 'RuntimeManager' = self.engine.RuntimeManager
        engineUBOName, runtimeUBOName, lightUBOName = runtimeManager.EngineUBOName, runtimeManager.RuntimeUBOName, runtimeManager.LightUBOName 
        engineBlockIndex = gl.glGetUniformBlockIndex(program, engineUBOName) # engine block contains engine settings, e.g. screen size
        
        if engineBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, engineBlockIndex, runtimeManager.EngineUBOBindingPoint)
        matrixBlockIndex = gl.glGetUniformBlockIndex(program, runtimeUBOName)
        
        if matrixBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, matrixBlockIndex, runtimeManager.RuntimeUBOBindingPoint)
        lightBlockIndex = gl.glGetUniformBlockIndex(program, lightUBOName)
        
        if lightBlockIndex != gl.GL_INVALID_INDEX:
            gl.glUniformBlockBinding(program, lightBlockIndex, runtimeManager.LightUBOBindingPoint)

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
    
    def setUniform(self, name: str, value:SupportedShaderValueTypes):
        if not valueTypeCheck(value, SupportedShaderValueTypes):
            raise TypeError(f'Invalid uniform value type: {type(value)}. Supported types: {get_args(SupportedShaderValueTypes)}')
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
        elif isinstance(value, ShaderStruct):
            # TODO: support struct
            raise NotImplementedError
        else:
            raise TypeError("Invalid uniform type: {}".format(type(value)))



__all__ = ['Shader','SupportedShaderValueTypes']