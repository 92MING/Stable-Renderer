from .manager import Manager
from runtime.gameObj import GameObject
import glfw

class RuntimeManager(Manager):
    '''Manager of GameObjs, Components, etc.'''

    _FrameEndFuncOrder = 999  # run at the end of each frame, since it need to count the frame time

    def __init__(self, fixedUpdateMaxFPS=60):
        super().__init__()
        self._fixedUpdateMaxFPS = fixedUpdateMaxFPS
        self._maxFixedUpdateDeltaTime = 1.0 / fixedUpdateMaxFPS
        self._deltaTime = 0.0
        self._startTime = 0.0

    @property
    def FixedUpdateMaxFPS(self):
        return self._fixedUpdateMaxFPS
    @FixedUpdateMaxFPS.setter
    def FixedUpdateMaxFPS(self, value):
        self._fixedUpdateMaxFPS = value
        self._maxFixedUpdateDeltaTime = 1.0 / value
    @property
    def DeltaTime(self):
        return self._deltaTime

    def _onFrameBegin(self):
        self._startTime = glfw.get_time()
    def _onFrameRun(self):
        if self.DeltaTime < self._maxFixedUpdateDeltaTime:
            GameObject._RunFixedUpdate()
        GameObject._RunUpdate()
        GameObject._RunLateUpdate()
    def _onFrameEnd(self):
        self._deltaTime = glfw.get_time() - self._startTime

__all__ = ['RuntimeManager']