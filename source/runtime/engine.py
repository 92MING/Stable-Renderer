import glfw
from utils.global_utils import *
from static.scene import *
import numpy as np
np.set_printoptions(suppress=True)
from .managers import *

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
    def __init__(self, scene: Scene = None, winTitle=None, winSize=(800, 480),):
        self._scene = scene
        if winTitle is not None:
            title = winTitle
        elif self._scene is not None:
            title = self._scene.name
        else:
            title = 'Stable Renderer'

        self._windowManager = WindowManager(title, winSize)
        self._inputManager = InputManager(self._windowManager.Window)
        self._runtimeManager = RuntimeManager()
        self._renderManager = RenderManager()
        self._sceneManager = SceneManager(self._scene)
        # endregion

    @property
    def WindowManager(self)->WindowManager:
        '''Window Manager. GLFW related.'''
        return self._windowManager
    @property
    def InputManager(self)->InputManager:
        '''Input info/ events...'''
        return self._inputManager
    @property
    def RuntimeManager(self)->RuntimeManager:
        '''Manager of GameObjs, Components, etc.'''
        return self._runtimeManager
    @property
    def RenderManager(self)->RenderManager:
        '''Manager of Renderers, Shaders, UBO, ... etc.'''
        return self._renderManager

    def prepare(self):
        '''You can override this method to do some prepare work'''
        raise NotImplementedError

    def _run_logic(self):
        pass
    def run(self):
        Manager._RunPrepare() # prepare work, mainly for sceneManager to build scene, load resources, etc.
        while not glfw.window_should_close(self.WindowManager.Window):
            Manager._RunFrameBegin() # input events, etc.
            Manager._RunFrameRun() # run logic / render, etc.
            Manager._RunFrameEnd() # swap buffers, etc.
        Manager._RunRelease()

    @classmethod
    def Run(cls):
        global _engine_singleton
        if _engine_singleton is None:
            _engine_singleton = cls()
        _engine_singleton.run()
