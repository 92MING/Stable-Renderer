from .manager import Manager
from runtime.gameObj import GameObject
import glfw
from static import Color
import OpenGL.GL as gl
import glm

class RuntimeManager(Manager):
    '''Manager of GameObjs, Components, etc.'''

    _FrameEndFuncOrder = 999  # run at the end of each frame, since it need to count the frame time

    def __init__(self,
                 fixedUpdateMaxFPS=60,
                 ambientLightCol:Color=Color.CLEAR,
                 ambientLightIntensity:float=0.1,
                 maxLightNum:int=256,):
        super().__init__()
        self._maxLightNum = maxLightNum
        self._fixedUpdateMaxFPS = fixedUpdateMaxFPS
        self._minFixedUpdateDeltaTime = 1.0 / fixedUpdateMaxFPS
        self._frame_count = 0
        self._deltaTime = 0.0
        self._startTime = 0.0
        self._firstFrame = True
        self._ambientLightCol = ambientLightCol
        self._ambientLightIntensity = ambientLightIntensity
        self._init_matrix_UBO()
        self._init_light_UBO()

    def _init_matrix_UBO(self):
        self._UBO_modelMatrix = glm.mat4(1.0)
        self._UBO_viewMatrix = glm.mat4(1.0)
        self._UBO_projectionMatrix = glm.mat4(1.0)
        self._UBO_MVP = glm.mat4(1.0)
        self._UBO_MVP_IT = glm.mat4(1.0)
        self._UBO_MV = glm.mat4(1.0)
        self._UBO_MV_IT = glm.mat4(1.0)
        self._UBO_cam_pos = glm.vec3(0.0, 0.0, 0.0) # world camera position
        self._UBO_cam_dir = glm.vec3(0.0, 0.0, 0.0) # world camera direction

        self._matrixUBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._matrixUBO)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4) + 2 * glm.sizeof(glm.vec3), None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.MatrixUBO_BindingPoint, self._matrixUBO)

        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_modelMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 1 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_viewMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 2 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_projectionMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MVP))  # MVP
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 4 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MVP_IT))  # MVP_IT
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 5 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MV))  # MV
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 6 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_MV_IT))  # MV_IT
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4), glm.sizeof(glm.vec3), glm.value_ptr(self._UBO_cam_pos))  # camera world position
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4) + glm.sizeof(glm.vec3), glm.sizeof(glm.vec3), glm.value_ptr(self._UBO_cam_dir))  # camera world forward direction
    def _init_light_UBO(self):
        self._lightUBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._lightUBO)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 2 * glm.sizeof(glm.vec4), None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.LightUBO_BindingPoint, self._lightUBO)

    # region UBO:Matrix
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
    @property
    def UBO_CamPos(self):
        return self._UBO_cam_pos
    @property
    def UBO_CamDir(self):
        return self._UBO_cam_dir
    def UpdateUBO_CamPos(self, camPos: glm.vec3):
        self._UBO_cam_pos = camPos
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4), glm.sizeof(glm.vec3),
                           glm.value_ptr(self._UBO_cam_pos))
    def UpdateUBO_CamDir(self, camDir: glm.vec3):
        self._UBO_cam_dir = camDir
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4) + glm.sizeof(glm.vec3),
                           glm.sizeof(glm.vec3), glm.value_ptr(self._UBO_cam_dir))
    def UpdateUBO_ModelMatrix(self, modelMatrix: glm.mat4):
        self._UBO_modelMatrix = modelMatrix
        self._UBO_MV = self._UBO_viewMatrix * self._UBO_modelMatrix
        self._UBO_MV_IT = glm.transpose(glm.inverse(self._UBO_MV))
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_MV
        self._UBO_MVP_IT = glm.transpose(glm.inverse(self._UBO_MVP))
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, glm.sizeof(glm.mat4), glm.value_ptr(self._UBO_modelMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 4 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP_IT))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 5 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MV))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 6 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MV_IT))
    def UpdateUBO_ViewMatrix(self, viewMatrix: glm.mat4):
        self._UBO_viewMatrix = viewMatrix
        self._UBO_MV = self._UBO_viewMatrix * self._UBO_modelMatrix
        self._UBO_MV_IT = glm.transpose(glm.inverse(self._UBO_MV))
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_MV
        self._UBO_MVP_IT = glm.transpose(glm.inverse(self._UBO_MVP))
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_viewMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 4 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP_IT))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 5 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MV))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 6 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MV_IT))
    def UpdateUBO_ProjMatrix(self, projectionMatrix: glm.mat4):
        self._UBO_projectionMatrix = projectionMatrix
        self._UBO_MVP = self._UBO_projectionMatrix * self._UBO_MV
        self._UBO_MVP_IT = glm.transpose(glm.inverse(self._UBO_MVP))
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.MatrixUBO)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 2 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_projectionMatrix))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 3 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP))
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 4 * glm.sizeof(glm.mat4), glm.sizeof(glm.mat4),
                           glm.value_ptr(self._UBO_MVP_IT))
    # endregion

    # region UBO: Light
    @property
    def LightUBO_BindingPoint(self):
        return 1
    # endregion

    # region Properties
    @property
    def MaxLightNum(self):
        return self._maxLightNum
    @property
    def AmbientLightCol(self):
        '''The color of ambient light.'''
        return self._ambientLightCol
    @AmbientLightCol.setter
    def AmbientLightCol(self, value):
        '''The color of ambient light.'''
        self._ambientLightCol = value
    @property
    def AmbientLightIntensity(self):
        '''The intensity of ambient light.'''
        return self._ambientLightIntensity
    @AmbientLightIntensity.setter
    def AmbientLightIntensity(self, value):
        '''The intensity of ambient light.'''
        self._ambientLightIntensity = value
    @property
    def FixedUpdateMaxFPS(self):
        return self._fixedUpdateMaxFPS
    @FixedUpdateMaxFPS.setter
    def FixedUpdateMaxFPS(self, value):
        self._fixedUpdateMaxFPS = value
        self._minFixedUpdateDeltaTime = 1.0 / value
    @property
    def DeltaTime(self):
        return self._deltaTime
    @property
    def FrameCount(self):
        return self._frame_count
    # endregion

    def _onFrameBegin(self):
        self._startTime = glfw.get_time()
    def _onFrameRun(self):
        if self._firstFrame or self.DeltaTime >= self._minFixedUpdateDeltaTime:
            GameObject._RunFixedUpdate()
        GameObject._RunUpdate()
        GameObject._RunLateUpdate()
        if self.DeltaTime >= self._minFixedUpdateDeltaTime:
            self._deltaTime = 0.0
        self._firstFrame = False
    def _onFrameEnd(self):
        self._deltaTime += (glfw.get_time() - self._startTime)
        self._frame_count += 1

__all__ = ['RuntimeManager']