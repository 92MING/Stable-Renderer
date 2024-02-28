import glm
import glfw
import OpenGL.GL as gl

from .manager import Manager
from ..runtime.gameObj import GameObject
from ..static import Color


class RuntimeManager(Manager):
    '''Manager of GameObjs, Components, etc.'''

    FrameEndFuncOrder = 999  # run at the end of each frame, since it need to count the frame time

    def __init__(self,
                 fixedUpdateMaxFPS=60,
                 ambientLightCol:Color=Color.CLEAR,
                 ambientLightIntensity:float=0.2,
                 gravity:glm.vec3=glm.vec3(0, -9.8, 0)
                 ):
        super().__init__()

        self._fixedUpdateMaxFPS = fixedUpdateMaxFPS
        self._minFixedUpdateDeltaTime = 1.0 / fixedUpdateMaxFPS
        self._frame_count = 0
        self._deltaTime = 0.0
        self._startTime = 0.0
        self._firstFrame = True
        self._gravity = gravity

        self._ambientLightCol = ambientLightCol
        self._ambientLightIntensity = ambientLightIntensity

        self._init_matrix_UBO()

    # region UBO:Matrix
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

    @property
    def MatrixUBO(self):
        return self._matrixUBO

    @property
    def MatrixUBO_BindingPoint(self):
        return self.engine.CreateOrGet_UBO_BindingPoint("MatrixUBO")

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

    # region UBO: Engine
    def _init_engine_UBO(self):
        self._engineUBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._engineUBO)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, glm.sizeof(glm.ivec2), None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.EngineUBO_BindingPoint, self._engineUBO)

    def Update_UBO_ScreenSize(self):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._engineUBO)
        self._screenSize = glm.ivec2(*self.engine.WindowManager.WindowSize())  # for saving pointer
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, glm.sizeof(glm.ivec2), glm.value_ptr(self._screenSize))

    @property
    def EngineUBO(self):
        return self._engineUBO

    @property
    def EngineUBO_BindingPoint(self):
        return self.engine.CreateOrGet_UBO_BindingPoint('EngineUBO')
    # endregion

    # region UBO: Light
    def _init_light_UBO(self):
        # TODO: finish this
        self._lightUBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._lightUBO)
        from ..runtime.components.light import Light
        for subLightType in Light.AllLightSubTypes():
            pass
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 7 * glm.sizeof(glm.mat4) + 2 * glm.sizeof(glm.vec3), None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.MatrixUBO_BindingPoint, self._lightUBO)

    @property
    def LightUBO(self):
        return self._lightUBO

    @property
    def LightUBO_BindingPoint(self):
        return self.engine.CreateOrGet_UBO_BindingPoint('LightUBO')
    # endregion

    # region Properties
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
        '''Delta time between this frame and the last frame.'''
        return self._deltaTime

    @property
    def FrameCount(self):
        return self._frame_count
    
    @property
    def Gravity(self):
        return self._gravity
    
    @Gravity.setter
    def Gravity(self, value:glm.vec3):
        self._gravity = value
    # endregion

    def on_frame_begin(self):
        self._startTime = glfw.get_time()

    def on_frame_run(self):
        if self._firstFrame or self.DeltaTime >= self._minFixedUpdateDeltaTime:
            GameObject._RunFixedUpdate()
        GameObject._RunUpdate()
        GameObject._RunLateUpdate()
        if self.DeltaTime >= self._minFixedUpdateDeltaTime:
            self._deltaTime = 0.0
        self._firstFrame = False

    def on_frame_end(self):
        self._deltaTime += (glfw.get_time() - self._startTime)
        self._frame_count += 1



__all__ = ['RuntimeManager']