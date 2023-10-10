import glm
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy

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
    block_color_buffer += [1.0, 1.0, 0.0, 1.0] * 8
    vertex_buffer = []
    for i in range(8):
        vertex_buffer += block_point_buffer[3 * i:3 * i + 3]
        vertex_buffer += block_color_buffer[4 * i:4 * i + 4]
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
    print(sys.getsizeof(a), a.nbytes)
    glBufferData(GL_ARRAY_BUFFER, a.nbytes, a, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    a = numpy.array(block_EBO_buffer, dtype='uint32')
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, a.nbytes, a, GL_STATIC_DRAW)
    block_EBO_buffer_len = len(block_EBO_buffer)

    glBindVertexArray(0)


rotate = [45, 0.0, 0]

def display(shader):
    glClear(GL_COLOR_BUFFER_BIT)

    glBindVertexArray(block_VAO)

    #projectionMatrix = Matrix.Perspective(45.0, 400 / 350, 0.1, 1000.0)
    projectionMatrix = glm.perspective(glm.radians(45.0), 400 / 350, 0.1, 1000.0)
    #viewMatrix = Matrix.LookAt([0.0, 0.0, 5], [0.0, 0.0, 0], Vector.up())
    viewMatrix = glm.lookAt(glm.vec3(0.0, 0.0, 5.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0, 1, 0))
    #modelMatrix = Matrix.Transformation([0.0, 0.0, 0], rotate, [1.0, 1.0, 1.0])
    modelMatrix = glm.mat4(1.0)
    modelMatrix = glm.scale(modelMatrix, glm.vec3(1, 1, 1))

    modelMatrix = glm.rotate(modelMatrix, glm.radians(rotate[2]), glm.vec3(0, 0, 1)) # z
    modelMatrix = glm.rotate(modelMatrix, glm.radians(rotate[0]), glm.vec3(1, 0, 0)) # x
    modelMatrix = glm.rotate(modelMatrix, glm.radians(rotate[1]), glm.vec3(0, 1, 0)) # y

    modelMatrix = glm.translate(modelMatrix, glm.vec3(0.0, 0.0, 0))
    #shader.setUniform('MVP', projectionMatrix * viewMatrix * modelMatrix)
    mvp = projectionMatrix * viewMatrix * modelMatrix
    glUniformMatrix4fv(shader._getUniformID("MVP"), 1, GL_FALSE, glm.value_ptr(mvp))
    glDrawElements(GL_QUADS, block_EBO_buffer_len, GL_UNSIGNED_INT, None)
    rotate[0] += 0.2
    rotate[1] += 0.2
    rotate[2] += 0.2
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
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
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
    # glutReshapeFunc(reshape)
    glutMainLoop()
