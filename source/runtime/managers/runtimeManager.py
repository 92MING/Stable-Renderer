from .manager import Manager
from runtime.gameObj import GameObject
import glfw

class RuntimeManager(Manager):
    '''Manager of GameObjs, Components, etc.'''

    _FrameEndFuncOrder = 999  # run at the end of each frame, since it need to count the frame time

    def __init__(self, fixedUpdateMaxFPS=60):
        super().__init__()
        self._fixedUpdateMaxFPS = fixedUpdateMaxFPS
        self._minFixedUpdateDeltaTime = 1.0 / fixedUpdateMaxFPS
        self._frame_count = 0
        self._deltaTime = 0.0
        self._startTime = 0.0
        self._firstFrame = True

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