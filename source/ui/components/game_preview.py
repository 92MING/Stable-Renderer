import OpenGL.GL as gl
import OpenGL.GLU as GLU
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.arrays import vbo

class GamePreview(QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Stable Renderer')
    
    def initializeGL(self):
        self.cubeVtxArray = np.array(
	    [[0.0, 0.0, 0.0],
	     [1.0, 0.0, 0.0],
	     [1.0, 1.0, 0.0],
	     [0.0, 1.0, 0.0],
	     [0.0, 0.0, 1.0],
	     [1.0, 0.0, 1.0],
	     [1.0, 1.0, 1.0],
	     [0.0, 1.0, 1.0]])
        self.vertVBO = vbo.VBO(np.reshape(self.cubeVtxArray,
                        (1, -1)).astype(np.float32))
        self.vertVBO.bind()
        
        self.cubeClrArray = np.array(
	    [[0.0, 0.0, 0.0],
	     [1.0, 0.0, 0.0],
	     [1.0, 1.0, 0.0],
	     [0.0, 1.0, 0.0],
	     [0.0, 0.0, 1.0],
	     [1.0, 0.0, 1.0],
	     [1.0, 1.0, 1.0],
	     [0.0, 1.0, 1.0 ]])
        self.colorVBO = vbo.VBO(np.reshape(self.cubeClrArray,
                        (1, -1)).astype(np.float32))
        self.colorVBO.bind()
        
        self.cubeIdxArray = np.array(
                [0, 1, 2, 3,
                3, 2, 6, 7,
                1, 0, 4, 5,
                2, 1, 5, 6,
                0, 3, 7, 4,
                7, 6, 5, 4 ]
            )
        
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def paintGL(self) -> None:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        gl.glPushMatrix()    # push the current matrix to the current stack

        gl.glTranslate(0.0, 0.0, -50.0)    # third, translate cube to specified depth
        gl.glScale(20.0, 20.0, 20.0)       # second, scale cube
        gl.glTranslate(-0.5, -0.5, -0.5)   # first, translate cube center to origin

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertVBO)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colorVBO)

        gl.glDrawElements(gl.GL_QUADS, len(self.cubeIdxArray), gl.GL_UNSIGNED_INT, self.cubeIdxArray)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()    # restore the previous modelview matrix


__all__ = ['GamePreview']