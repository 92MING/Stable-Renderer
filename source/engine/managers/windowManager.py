import glfw
import OpenGL.GL as gl

from typing import Tuple, TYPE_CHECKING

from .manager import Manager
from .renderManager import RenderManager
from common_utils.data_struct import DelayEvent, Event
from common_utils.constants import NAME, VERSION
from common_utils.global_utils import is_editor_mode
from engine.static import Color

if TYPE_CHECKING:
    from ui.components import GamePreview
    from ui.main import MainWindow


class WindowManager(Manager):

    FrameEndFuncOrder = RenderManager.FrameEndFuncOrder + 1 # swap buffer should be called after render
    ReleaseFuncOrder = 999 # terminate glfw should be called at the end

    _glfw_window: int
    _pyqt_window: 'GamePreview'
    _pyqt_main_window: 'MainWindow'

    def __init__(self, 
                 title: str|None, 
                 size: tuple[int, int], 
                 windowResizable: bool = False, 
                 bgColor=Color.CLEAR):
        super().__init__()
        
        self._bgColor = bgColor
        self._title = title or f"{NAME} v{VERSION}"
        self._size = size
        self._resizable = windowResizable
        
        if is_editor_mode():
            self._init_pyqt()
        else:
            self._init_glfw()
        self._onWindowResize = DelayEvent(int, int)
        self._onWindowResize.addListener(lambda width, height: gl.glViewport(0, 0, width, height))
        gl.glClearColor(bgColor.r, bgColor.g, bgColor.b, bgColor.a)

    def _init_glfw(self):
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
        self._glfw_window = glfw.create_window(self._size[0], self._size[1], self._title, None, None)
        if not self._glfw_window:
            glfw.terminate()
            exit()
        glfw.make_context_current(self._glfw_window)
        glfw.set_window_attrib(self._glfw_window, glfw.RESIZABLE, self._resizable)
        glfw.set_window_size_callback(self._glfw_window, self._resizeCallback)
    
    def _init_pyqt(self):
        from ui.components import GamePreview
        from ui.main import MainWindow
        if not hasattr(MainWindow, '__instance__') or not MainWindow.__instance__:    # type: ignore
            raise ValueError("Editor UI is not yet initialized. For running editor mode, please run engine through ui's entry point.")
        if not hasattr(GamePreview, '__instance__') or not GamePreview.__instance__:    # type: ignore
            raise ValueError("Editor UI is not yet initialized. For running editor mode, please run engine through ui's entry point.")
        self._pyqt_window: GamePreview = GamePreview.__instance__ # type: ignore
        self._pyqt_main_window: MainWindow = MainWindow.__instance__     # type: ignore
        self._title = self._pyqt_main_window.title_bar.title.text()
        
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
        glfw.set_window_attrib(self._glfw_window, glfw.RESIZABLE, value)
    
    @property
    def OnWindowResize(self)->Event:
        return self._onWindowResize
    
    @property
    def Window(self):
        return self._glfw_window
    
    @property
    def Title(self):
        return self._title
    
    @Title.setter
    def Title(self, value):
        self.SetWindowTitle(value)
    
    def SetWindowTitle(self, title):
        self._title = title
        glfw.set_window_title(self._glfw_window, title)
    
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
        glfw.set_window_size(self._glfw_window, *self._size)
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
            glfw.show_window(self._glfw_window)
        else:
            glfw.hide_window(self._glfw_window)
    
    # endregion


__all__ = ['WindowManager']