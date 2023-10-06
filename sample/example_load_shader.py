from __future__ import division
from OpenGL.GL import *
import textwrap


vertex_shader_source = textwrap.dedent("""\
    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;

    attribute vec3 aVertex;
    attribute vec3 aNormal;
    attribute vec2 aTexCoord;

    varying vec2 vTexCoord;

    void main(){
       vTexCoord = aTexCoord;
       // Make GL think we are actually using the normal
       aNormal;
       gl_Position = (uPMatrix * uMVMatrix)  * vec4(aVertex, 1.0);
    }
    """)

fragment_shader_source = textwrap.dedent("""\
    uniform sampler2D sTexture;
    varying vec2 vTexCoord;
    void main(){
       gl_FragColor = texture2D(sTexture, vTexCoord);
    }
    """)


def load_program(vertex_source, fragment_source):
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source)
    if vertex_shader == 0:
        return 0

    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source)
    if fragment_shader == 0:
        return 0

    program = glCreateProgram()

    if program == 0:
        return 0

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        glDeleteProgram(program)
        return 0

    return program

def load_shader(shader_type, source):
    shader = glCreateShader(shader_type)

    if shader == 0:
        return 0

    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS, None) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(info_log)
        glDeleteProgram(shader)
        return 0

    return shader
