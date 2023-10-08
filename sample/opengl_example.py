from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy
from static.data_types.vector import Vector
from static.data_types.matrix import Matrix

block_VAO = 0
draw = False
block_EBO_buffer_len = 0

def create_blocks(x: int, y: int, z: int):
    global draw, block_VAO, block_EBO_buffer_len
    if draw:
        return
    draw = True
    block_point_buffer = []
    block_color_buffer = []
    block_EBO_buffer = []
    block_point_buffer += [x - 0.5, y + 0.5, z - 0.5,  # V0
                           x + 0.5, y + 0.5, z - 0.5,  # V1
                           x + 0.5, y - 0.5, z - 0.5,  # V2
                           x - 0.5, y - 0.5, z - 0.5,  # V3
                           x - 0.5, y + 0.5, z + 0.5,  # V4
                           x + 0.5, y + 0.5, z + 0.5,  # V5
                           x + 0.5, y - 0.5, z + 0.5,  # V6
                           x - 0.5, y - 0.5, z + 0.5]  # V7
    block_color_buffer += [1.0, 0.0, 1.0, 1.0] * 8
    vertex_buffer = []
    for i in range(len(block_point_buffer)//3):
        vertex_buffer += block_point_buffer[3*i:3*i+3]
        vertex_buffer += block_color_buffer[4*i:4*i+4]
    block_EBO_buffer += [0, 1, 5, 4,
                         3, 2, 6, 7,
                         0, 3, 7, 4,
                         1, 2, 6, 5,
                         0, 1, 2, 3,
                         4, 5, 6, 7]

    block_VAO = glGenVertexArrays(1)
    glBindVertexArray(block_VAO)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    a = numpy.array(vertex_buffer, dtype='float32')
    glBufferData(GL_ARRAY_BUFFER, sys.getsizeof(a), a, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*4, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*4, ctypes.c_void_p(3*4))

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    a = numpy.array(block_EBO_buffer, dtype='uint32')
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sys.getsizeof(a), a, GL_STATIC_DRAW)
    block_EBO_buffer_len = len(a)

    glBindVertexArray(0)

rotate = [33.0, 40.0, 20.0]
def display(shader):
    glMatrixMode(GL_MODELVIEW)
    glClear(GL_COLOR_BUFFER_BIT)

    modelMatrix = Matrix.Transformation([0.0, 0.0, -4.5], rotate, [1.0, 1.0, 1.0])
    viewMatrix = Matrix.LookAt([0.0, 0.0, 0.0], [0.0, 0.0, -20], Vector.up())
    projectionMatrix = Matrix.Perspective(45.0, 400/350, 0.5, 20.0)
    shader.setUniform('MVP', projectionMatrix * viewMatrix * modelMatrix)

    glBindVertexArray(block_VAO)
    glDrawElements(GL_QUADS, block_EBO_buffer_len, GL_UNSIGNED_INT, None)

    rotate[1] += 10.0

    glutSwapBuffers()
    glutPostRedisplay()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40.0, width / height, 0.5, 20.0)
    glMatrixMode(GL_MODELVIEW)

if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    glutInitWindowSize(400, 350)
    glutCreateWindow(b"OpenGL Window")
    create_blocks(0, 0, 0)
    glClearColor(0.0, 0.0, 0.0, 0.0)

    from static.shader import Shader
    shader = Shader('test', 'vShader.glsl', 'fShader.glsl')
    shader.useProgram()

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    def _display():
        display(shader)
    glutDisplayFunc(_display)
    #glutReshapeFunc(reshape)
    glutMainLoop()