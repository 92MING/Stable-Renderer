from .manager import Manager
from utils.data_struct import DelayEvent, Event
import glfw
import OpenGL.GL as gl

class WindowManager(Manager):
    def __init__(self, title, size):
        super().__init__()
        self._init_glfw(title, size)
        self._onWindowResize = DelayEvent()
        self._onWindowResize.addListener(lambda width, height: gl.glViewport(0, 0, width, height))
    def _init_glfw(self, winTitle, winSize):
        self._title = winTitle
        self._size = winSize
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, gl.GL_TRUE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.SAMPLES, 4)  # MSAA
        glfw.window_hint(glfw.STENCIL_BITS, 8)
        self._window = glfw.create_window(winSize[0], winSize[1], winTitle, None, None)
        if not self._window:
            glfw.terminate()
            exit()
        glfw.make_context_current(self._window)
        glfw.set_window_size_callback(self._window, self._resizeCallback)
    def _resizeCallback(self, window, width, height):
        self._onWindowResize.invoke(width, height)

    def _onFrameBegin(self):
        self._onWindowResize.release()
    def _release(self):
        glfw.terminate()

    @property
    def OnWindowResize(self)->Event:
        return self._onWindowResize
    @property
    def Window(self):
        return self._window
    @property
    def Title(self):
        return self._title
    @Title.setter
    def Title(self, value):
        self.SetWindowTitle(value)
    def SetWindowTitle(self, title):
        self._title = title
        glfw.set_window_title(self._window, title)
    @property
    def WindowSize(self):
        return self._size
    @WindowSize.setter
    def WindowSize(self, value):
        self.SetWindowSize(*value)
    def SetWindowSize(self, width, height):
        self._size = (width, height)
        glfw.set_window_size(self._window, *self._size)
    @property
    def AspectRatio(self):
        return self._size[0] / self._size[1]
__all__ = ['WindowManager']