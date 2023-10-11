from static.color import *
from static.enums import *
import glfw
from utils.data_struct import Event
from utils.global_utils import *
from static.scene import *
import numpy as np
np.set_printoptions(suppress=True)

class DelayEvent(Event):
    '''
    When invoke delay event, it will not be executed immediately, but params will be save and execute later by calling "release".
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_param = None

    def invoke(self, *args):
        self._check_invoke_params(*args)
        self._last_param = args

    def release(self):
        if self._last_param is not None:
            super().invoke(*self._last_param)
        self._last_param = None
class TaskList:
    def __init__(self):
        self._tasks = set()

    def addTask(self, task):
        self._tasks.add(task)

    def removeTask(self, task):
        self._tasks.remove(task)

    def execute(self):
        for task in self._tasks:
            task()
        self._tasks.clear()

_engine_singleton = GetOrAddGlobalValue("_ENGINE_SINGLETON", None)
class Engine:

    def __new__(cls, *args, **kwargs):
        global _engine_singleton
        if _engine_singleton is not None:
            clsName = _engine_singleton.__class__.__qualname__
            if clsName != cls.__qualname__:
                raise RuntimeError(f'Engine is running as {clsName}. As a singleton, it can not be created again.')
            _engine_singleton.__init__ = lambda *a, **kw: None  # 防止再次初始化
            return _engine_singleton
        else:
            e = super().__new__(cls)
            SetGlobalValue("_ENGINE_SINGLETON", e)
            _engine_singleton = e
            return e

    def __init__(self,
                 scene: Scene = None,
                 winTitle=None,
                 winSize=(800, 480),
                 bgColor=Color.CLEAR, ):

        self._scene = scene
        if winTitle is not None:
            title = winTitle
        elif self._scene is not None:
            title = self._scene.name
        else:
            title = 'Stable Renderer'
        self._winTitle = title
        self._winSize = winSize
        self._bgColor = bgColor
        self._init_glfw(self._winTitle, self._winSize)

        self._onNextLoopStart = TaskList()
        self._renderTasks = TaskList()
        self._onNextLoopStart.addTask(lambda: gl.glViewport(0, 0, *self._winSize))
        self._onNextLoopStart.addTask(lambda: gl.glClearColor(bgColor.r, bgColor.g, bgColor.b, bgColor.a))
        self._onNextLoopStart.addTask(lambda: gl.glEnable(gl.GL_DEPTH_TEST))
        self._onNextLoopStart.addTask(lambda: gl.glEnable(gl.GL_CULL_FACE))

        self._init_callbacks()

    def _init_glfw(self, winTitle, winSize):
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
        self.window = glfw.create_window(winSize[0], winSize[1], winTitle, None, None)
        if not self.window:
            glfw.terminate()
            exit()
        glfw.make_context_current(self.window)
    def _init_callbacks(self):
        self.window_resize_event = DelayEvent(int, int)

        def resizeCallback(window, width, height):
            self._onNextLoopStart.addTask(lambda: gl.glViewport(0, 0, width, height))
            self.window_resize_event.invoke(width, height)

        glfw.set_window_size_callback(self.window, resizeCallback)

        glfw.set_key_callback(self.window,
                              lambda window, key, scancode, action, mods: print(key, scancode, action, mods))
        glfw.set_mouse_button_callback(self.window, lambda window, button, action, mods: print(button, action, mods))
        glfw.set_cursor_pos_callback(self.window, lambda window, xpos, ypos: print(xpos, ypos))
        glfw.set_framebuffer_size_callback(self.window, lambda window, width, height: print(width, height))
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS, 1)

    # region properties
    @property
    def winTitle(self):
        return self._winTitle

    @property
    def winSize(self):
        return self._winSize

    @winSize.setter
    def winSize(self, value):
        self._winSize = value
        self._onNextLoopStart.addTask(lambda: glfw.set_window_size(self.window, *self._winSize))

    @property
    def bgColor(self):
        return self._bgColor

    @bgColor.setter
    def bgColor(self, value):
        self._bgColor = value
        self._onNextLoopStart.addTask(
            lambda: gl.glClearColor(self._bgColor.r, self._bgColor.g, self._bgColor.b, self._bgColor.a))

    # endregion

    def addRenderTask(self, task):
        self._renderTasks.addTask(task)

    def prepare(self):
        '''You can override this method to do some prepare work'''
        raise NotImplementedError

    def _prepare(self):
        '''real prepare method'''
        if self._scene is not None:
            self._scene.prepare()
        else:
            try:
                self.prepare()
            except NotImplementedError:
                raise Exception('You must override "prepare" method or set "scene" to prepare data.')

    def _run_logic(self):
        pass

    def run(self):
        self._prepare()
        while not glfw.window_should_close(self.window):

            # region before run logic
            self._onNextLoopStart.execute() # some system tasks, e.g. glClearColor, glViewport

            # handle events (release all late events)
            glfw.poll_events()
            self.window_resize_event.release()
            # endregion

            # region run logic
            self._run_logic()
            # endregion

            # region render
            self._renderTasks.execute()
            # endregion

            # end loop
            glfw.swap_buffers(self.window)

        glfw.terminate()

    @classmethod
    def Run(cls):
        global _engine_singleton
        if _engine_singleton is None:
            _engine_singleton = cls()
        _engine_singleton.run()
