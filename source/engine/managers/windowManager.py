import glfw
import OpenGL.GL as gl

from .manager import Manager
from .renderManager import RenderManager
from common_utils.data_struct import DelayEvent, Event
from typing import Tuple
from ..static import Color



class WindowManager(Manager):

    FrameEndFuncOrder = RenderManager.FrameEndFuncOrder + 1 # swap buffer should be called after render
    ReleaseFuncOrder = 999 # terminate glfw should be called at the end

    def __init__(self, title, size, windowResizable = False, bgColor=Color.CLEAR):
        super().__init__()
        self._bgColor = bgColor
        self._init_glfw(title, size, windowResizable)
        self._onWindowResize = DelayEvent(int, int)
        self._onWindowResize.addListener(lambda width, height: gl.glViewport(0, 0, width, height))
        gl.glClearColor(bgColor.r, bgColor.g, bgColor.b, bgColor.a)

    def _init_glfw(self, winTitle, winSize, winResizable):
        self._title = winTitle
        self._size = winSize
        self._resizable = winResizable
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
        glfw.set_window_attrib(self._window, glfw.RESIZABLE, winResizable)
        glfw.set_window_size_callback(self._window, self._resizeCallback)
    
    def _resizeCallback(self, window, width, height):
        self._onWindowResize.invoke(width, height)

    def on_frame_begin(self):
        self._onWindowResize.release()
    
    def release(self):
        glfw.terminate()

    # region properties
    @property
    def BgColor(self):
        return self._bgColor
    
    @BgColor.setter
    def BgColor(self, value:Color):
        self._bgColor = value
        gl.glClearColor(value.r, value.g, value.b, value.a)
    
    @property
    def WindowResizable(self):
        return self._resizable
    
    @WindowResizable.setter
    def WindowResizable(self, value):
        self._resizable = value
        glfw.set_window_attrib(self._window, glfw.RESIZABLE, value)
    
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
        '''(width, height)'''
        return self._size
    
    @WindowSize.setter
    def WindowSize(self, value: Tuple[int, int]):
        '''(width, height)'''
        self.SetWindowSize(*value)
    
    def SetWindowSize(self, width: int, height: int):
        self._size = (width, height)
        glfw.set_window_size(self._window, *self._size)
        self.engine.RuntimeManager.Update_UBO_ScreenSize() # update UBO data
    
    @property
    def WindowWidth(self):
        return self._size[0]
    
    @property
    def WindowHeight(self):
        return self._size[1]
    
    @property
    def AspectRatio(self):
        return self._size[0] / self._size[1]
    # endregion

    # region public methods
    def SetWindowVisible(self, visible:bool):
        if visible:
            glfw.show_window(self._window)
        else:
            glfw.hide_window(self._window)
    
    # endregion


__all__ = ['WindowManager']