import glfw
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu

from colorama import Fore, Style
from enum import Enum
from typing import Optional

from utils.global_utils import *
from utils.decorators import class_or_ins_property, prevent_re_init 
from utils.data_struct import Event
from .managers import *
from .static.scene import *
from .static import Color


class EngineStage(Enum):
    NOT_YET_START = 0
    PREPARE = 1
    FRAME_BEGIN = 2
    FRAME_RUN = 3
    FRAME_END = 4
    RELEASE = 5

np.set_printoptions(suppress=True)

_EngineInstance: Optional['Engine'] = GetOrAddGlobalValue("_ENGINE_SINGLETON", None)

_OnBeforeStart = Event()
_OnBeforeEnd = Event()

@prevent_re_init
class Engine:
    '''
    The base class of engine. You can inherit this class to create your own engine.
    Create and run the engine by calling `Engine.Run()`. It will create a singleton instance of the engine and run it.
    '''

    @class_or_ins_property
    def OnBeforeStart(cls_or_self)->Event:
        return _OnBeforeStart
    
    @class_or_ins_property
    def OnBeforeEnd(cls_or_self)->Event:
        return _OnBeforeEnd
    
    @class_or_ins_property
    def IsRunning(cls_or_self)->bool:
        '''Whether the engine is in running stage(FrameBegin, FrameRun, FrameEnd).'''
        if not _EngineInstance:
            return False
        return _EngineInstance._stage in (EngineStage.FRAME_BEGIN, EngineStage.FRAME_RUN, EngineStage.FRAME_END)
    
    def __new__(cls, *args, **kwargs):
        global _EngineInstance
        if _EngineInstance is not None:
            clsName = _EngineInstance.__class__.__qualname__
            if clsName != cls.__qualname__:
                raise RuntimeError(f'Engine is running as {clsName}. As a singleton, it can not be created again.')
            return _EngineInstance
        else:
            e = super().__new__(cls)
            SetGlobalValue("_ENGINE_SINGLETON", e)
            _EngineInstance = e
            return e

    def __init__(self,
                 scene: Optional[Scene] = None,
                 winTitle=None,
                 winSize=(1080, 720),
                 windowResizable=False,
                 bgColor=Color.CLEAR,
                 enableHDR=True,
                 enableGammaCorrection=True,
                 gamma=2.2,
                 exposure=1.0,
                 saturation=1.0,
                 brightness=1.0,
                 contrast=1.0,
                 debug=False,
                 needOutputMaps=False,
                 mapMinimizeRatio = 64,
                 maxFrameCacheCount=24,
                 mapSavingInterval=12,
                 threadPoolSize=6,
                 target_device: Optional[int]=None,):

        self.AcceptedPrint('Engine is initializing...')
        self._UBO_Binding_Points = {}
        self._debug = debug
        self._scene = scene
        if winTitle is not None:
            title = winTitle
        elif self._scene is not None:
            title = self._scene.name
        else:
            title = 'Stable Renderer'

        # region managers
        self._windowManager = WindowManager(title, winSize, windowResizable, bgColor)
        self._inputManager = InputManager(self._windowManager.Window)
        self._runtimeManager = RuntimeManager()
        self._renderManager = RenderManager(enableHDR=enableHDR, 
                                            enableGammaCorrection=enableGammaCorrection, 
                                            gamma=gamma, 
                                            exposure=exposure,
                                            saturation=saturation, 
                                            brightness=brightness, 
                                            contrast=contrast,
                                            target_device=target_device)
        self._diffusionManager = DiffusionManager(needOutputMaps=needOutputMaps,
                                           maxFrameCacheCount=maxFrameCacheCount,
                                           mapSavingInterval=mapSavingInterval,
                                           threadPoolSize=threadPoolSize,
                                           mapMinimizeRatio=mapMinimizeRatio)
        self._sceneManager = SceneManager(self._scene)
        self._resourceManager = ResourcesManager()
        
        # endregion
        
        self._stage = EngineStage.NOT_YET_START

    @property
    def Stage(self)->EngineStage:
        return self._stage

    # region debug
    @property
    def IsDebugMode(self):
        return self._debug
    
    @IsDebugMode.setter
    def IsDebugMode(self, value):
        self._debug = value

    def PrintOpenGLError(self):
        try:
            gl.glGetError() # nothing to do with error, just clear error flag
        except Exception as e:
            print('GL ERROR: ', glu.gluErrorString(e))
            
    def WarningPrint(self, *args, **kwargs):
        '''Print as yellow and bold text.'''
        print(Fore.YELLOW + Style.BRIGHT, end='')
        print(*args, **kwargs)
        print(Style.RESET_ALL, end='')
    
    def ErrorPrint(self, *args, **kwargs):
        '''Print as red and bold text.'''
        print(Fore.RED + Style.BRIGHT, end='')
        print(*args, **kwargs)
        print(Style.RESET_ALL, end='')
    
    def InfoPrint(self, *args, **kwargs):
        '''Print as blue and bold text.'''
        print(Fore.BLUE + Style.BRIGHT, end='')
        print(*args, **kwargs)
        print(Style.RESET_ALL, end='')
    
    def AcceptedPrint(self, *args, **kwargs):
        '''Print as green and bold text.'''
        print(Fore.GREEN + Style.BRIGHT, end='')
        print(*args, **kwargs)
        print(Style.RESET_ALL, end='')
    # endregion

    # region managers
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
    
    @property
    def SceneManager(self)->SceneManager:
        return self._sceneManager
    
    @property
    def ResourcesManager(self)->ResourcesManager:
        return self._resourceManager
    
    @property
    def DiffusionManager(self)->DiffusionManager:
        return self._diffusionManager
    # endregion

    def CreateOrGet_UBO_BindingPoint(self, name:str):
        if name not in self._UBO_Binding_Points:
            self._UBO_Binding_Points[name] = len(self._UBO_Binding_Points)
        return self._UBO_Binding_Points[name]

    # region overridable methods （for debug & easy customize)
    def beforePrepare(self):...
    def afterPrepare(self):...
    def beforeFrameBegin(self):...
    def beforeFrameRun(self):...
    def beforeFrameEnd(self):...
    def beforeRelease(self):...
    def afterRelease(self):...
    # endregion

    def run(self):
        self.OnBeforeStart.invoke()
        
        self.AcceptedPrint('Engine Preparing...')
        
        self.beforePrepare()
        Manager._RunPrepare()  # prepare work, mainly for sceneManager to build scene, load resources, etc.
        self.afterPrepare()

        self.AcceptedPrint('Engine Preparation Done.')
        self.AcceptedPrint('Engine Start Running...')
        while not glfw.window_should_close(self.WindowManager.Window):

            self.beforeFrameBegin()
            Manager._RunFrameBegin()  # input events / clear buffers / etc.

            self.beforeFrameRun()
            Manager._RunFrameRun() # update gameobjs, components / render / ... etc.

            self.beforeFrameEnd()
            Manager._RunFrameEnd() #  swap buffers / time count / etc.

        self.AcceptedPrint('Engine is ending...')
        self.AcceptedPrint('Releasing Resources...')
        self.beforeRelease()
        Manager._RunRelease()
        self.afterRelease()
        
        self.OnBeforeEnd.invoke()
        
        self.AcceptedPrint('Engine is ended.')

    @classmethod
    def Run(cls, *args, **kwargs):
        global _EngineInstance
        if _EngineInstance is None:
            _EngineInstance = cls(*args, **kwargs)
        _EngineInstance.run()