from .manager import Manager
from .runtimeManager import RuntimeManager
from utils.data_struct.event import AutoSortTask
import glm
import OpenGL.GL as gl
import OpenGL.GLU as glu

class RenderManager(Manager):
    '''Manager of all rendering stuffs'''

    _FrameRunFuncOrder = RuntimeManager._FrameRunFuncOrder + 1  # always run after runtimeManager

    def __init__(self):
        super().__init__()
        self._renderTasks = AutoSortTask()
        self._init_opengl()
    def _init_opengl(self):
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        self._UBO_modelMatrix = glm.mat4(1.0)
        self._UBO_viewMatrix = glm.mat4(1.0)
        self._UBO_projectionMatrix = glm.mat4(1.0)
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_viewMatrix * self._UBO_modelMatrix

        self._matrixUBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._matrixUBO)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 4 * glm.sizeof(glm.mat4), None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.MatrixUBO_BindingPoint, self._matrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_modelMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                        glm.value_ptr(self._UBO_viewMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 2 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                        glm.value_ptr(self._UBO_projectionMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                        glm.value_ptr(self._UBO_MVP))  # MVP

    # region UBO
    @property
    def MatrixUBO(self):
        return self._matrixUBO
    @property
    def MatrixUBO_BindingPoint(self):
        return 0
    @property
    def UBO_ModelMatrix(self):
        return self._UBO_modelMatrix
    @property
    def UBO_ViewMatrix(self):
        return self._UBO_viewMatrix
    @property
    def UBO_ProjMatrix(self):
        return self._UBO_projectionMatrix
    def UpdateUBO_ModelMatrix(self, modelMatrix: glm.mat4):
        self._UBO_modelMatrix = modelMatrix
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_viewMatrix * self._UBO_modelMatrix
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_modelMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MVP))
    def UpdateUBO_ViewMatrix(self, viewMatrix: glm.mat4):
        self._UBO_viewMatrix = viewMatrix
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_viewMatrix * self._UBO_modelMatrix
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),  glm.value_ptr(self._UBO_viewMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MVP))
    def UpdateUBO_ProjMatrix(self, projectionMatrix: glm.mat4):
        self._UBO_projectionMatrix = projectionMatrix
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_viewMatrix * self._UBO_modelMatrix
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 2 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),  glm.value_ptr(self._UBO_projectionMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MVP))
    def printOpenGLError(self):
        try:
            gl.glGetError() # nothing to do with error, just clear error flag
        except Exception as e:
            print('GL ERROR: ', e)
    # endregion

    @property
    def RenderTasks(self):
        return self._renderTasks

    def _onFrameBegin(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    def _onFrameRun(self):
        self._renderTasks.execute()


__all__ = ['RenderManager']