import os
import sys
import glfw
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu

from common_utils.path_utils import SOURCE_DIR

if str(SOURCE_DIR.absolute()) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR.absolute()))

_COMFYUI_PATH = str((SOURCE_DIR / 'comfyUI').absolute())
if _COMFYUI_PATH not in sys.path:
    sys.path.insert(0, _COMFYUI_PATH)

#! DONT DELETE THE FOLLOWING 2 LINES ! IT IS FOR CUDA INITIALIZATION 
import comfyUI.nodes
import pycuda.autoprimaryctx

from enum import Enum
from typing import Optional, Union
from inspect import signature
from pathlib import Path
from common_utils.debug_utils import EngineLogger
from common_utils.global_utils import GetOrAddGlobalValue, GetOrCreateGlobalValue, SetGlobalValue, GetGlobalValue, is_dev_mode
from common_utils.decorators import class_or_ins_property, prevent_re_init 
from common_utils.data_struct import Event
from common_utils.cuda_utils import get_cuda_device
from .managers import *
from .static.enums import EngineMode, EngineStage
from .static.scene import Scene
from .static import Color, Workflow

if is_dev_mode():
    np.set_printoptions(suppress=True)



_EngineInstance: Optional['Engine'] = GetOrAddGlobalValue("_ENGINE_SINGLETON", None)    # type: ignore
_OnEngineStageChanged: Event = GetOrCreateGlobalValue("_ON_ENGINE_STAGE_CHANGED", Event, EngineStage)

@prevent_re_init
class Engine:
    '''
    The base class of engine. You can inherit this class to create your own engine.
    Create and run the engine by calling `Engine.Run()`. It will create a singleton instance of the engine and run it.
    '''
    
    @staticmethod
    def Instance():
        if not _EngineInstance:
            raise RuntimeError('Engine is not initialized yet.')
        return _EngineInstance
    
    @class_or_ins_property  # type: ignore
    def IsLooping(cls_or_self)->bool:
        '''Whether the engine is within running stage(FrameBegin, FrameRun, FrameEnd).'''
        if not _EngineInstance:
            return False
        return _EngineInstance._stage in EngineStage.RunningStages()
    
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
                 enableGammaCorrection=False,
                 gamma=2.2,
                 exposure=1.0,
                 saturation=1.0,
                 brightness=1.0,
                 contrast=1.0,
                 debug=False,
                 needOutputMaps=False,
                 maxFrameCacheCount=24,
                 mapSavingInterval=12,
                 threadPoolSize=6,
                 target_device: Optional[int]=None,
                 mode: EngineMode=EngineMode.GAME,
                 disableComfyUI: bool = False,
                 workflow: Union[Workflow, str, Path, None] = None,
                 **kwargs):
        
        if disableComfyUI:
            EngineLogger.warn('ComfyUI is disable. This setting is just for debugging. Please make sure you are not in production mode.')
            self.disableComfyUI = True
        else:
            self.disableComfyUI = False
        self._mode = mode
        EngineLogger.info(f'Engine start with {mode.name} mode. Initializing...')
        
        if not self.disableComfyUI:
            from comfyUI.main import run
            self._prompt_executor = run()

        if target_device is None:
            target_device = get_cuda_device()
        self._target_device = target_device
        
        self._UBO_Binding_Points = {}
        self._debug = debug
        
        if winTitle is not None:
            title = winTitle
        elif scene is not None:
            title = scene.name
        else:
            title = 'Stable Renderer'

        # region managers
        def find_kwargs_for_manager(manager):
            init_sig = signature(manager.__init__)
            found_args = {}
            for arg_name, arg in kwargs.items():
                if arg_name in init_sig.parameters:
                    found_args[arg_name] = arg  # will not pop from kwargs, i.e. when managers have same arg name, it will be passed to all of them
            return found_args
        
        self._windowManager = WindowManager(title=title, 
                                            size=winSize, 
                                            windowResizable=windowResizable, 
                                            bgColor=bgColor, 
                                            **find_kwargs_for_manager(WindowManager))
        
        self._inputManager = InputManager(window = self._windowManager.Window, **find_kwargs_for_manager(InputManager))
        
        self._runtimeManager = RuntimeManager(**find_kwargs_for_manager(RuntimeManager))
        
        self._renderManager = RenderManager(enableHDR=enableHDR, 
                                            enableGammaCorrection=enableGammaCorrection, 
                                            gamma=gamma, 
                                            exposure=exposure,
                                            saturation=saturation, 
                                            brightness=brightness, 
                                            contrast=contrast,
                                            **find_kwargs_for_manager(RenderManager))
        
        self._diffusionManager = DiffusionManager(needOutputMaps=needOutputMaps,
                                                  maxFrameCacheCount=maxFrameCacheCount,
                                                  mapSavingInterval=mapSavingInterval,
                                                  threadPoolSize=threadPoolSize,
                                                  workflow=workflow,
                                                  **find_kwargs_for_manager(DiffusionManager))
        
        self._sceneManager = SceneManager(mainScene=scene, **find_kwargs_for_manager(SceneManager))
        
        self._resourceManager = ResourcesManager(**find_kwargs_for_manager(ResourcesManager))
        # endregion
        
        self._stage = EngineStage.INIT
        _OnEngineStageChanged.invoke(self._stage)
    
    @property
    def OnStageChanged(self)->Event:
        return _OnEngineStageChanged
    
    @property
    def Mode(self):
        '''Running mode of the engine. It can be 'game' or 'editor'. Default is 'game'.'''
        return self._mode
    
    @property
    def TargetDevice(self)->int:
        '''The target device for cuda. Default is 0.'''
        return self._target_device

    @property
    def PromptExecutor(self):
        if not hasattr(self, '_prompt_executor'):
            raise AttributeError('PromptExecutor is not available. Please set disableComfyUI=False when initializing the engine.')
        return self._prompt_executor

    @property
    def Stage(self)->EngineStage:
        return self._stage
    
    @Stage.setter
    def Stage(self, value:EngineStage):
        if value == self._stage:
            return
        self._stage = value
        _OnEngineStageChanged.invoke(value)

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
            EngineLogger.error('GL ERROR: ', glu.gluErrorString(e))
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

    def Pause(self):
        raise NotImplementedError # TODO
    
    def Continue(self):
        raise NotImplementedError # TODO
    
    def run(self):        
        EngineLogger.info('Engine Preparing...')
        
        self.Stage = EngineStage.BEFORE_PREPARE
        self.beforePrepare()
        Manager._RunPrepare()  # prepare work, mainly for sceneManager to build scene, load resources, etc.
        self.afterPrepare()
        self.Stage = EngineStage.AFTER_PREPARE

        EngineLogger.info('Engine Preparation Done. Start Running...')
        while not glfw.window_should_close(self.WindowManager.Window):
            
            self.Stage = EngineStage.BEFORE_FRAME_BEGIN
            self.beforeFrameBegin()
            Manager._RunFrameBegin()  # input events / clear buffers / etc.

            self.Stage = EngineStage.BEFORE_FRAME_RUN
            self.beforeFrameRun()
            Manager._RunFrameRun() # update gameobjs, components / render / ... etc.

            self.Stage = EngineStage.BEFORE_FRAME_END
            self.beforeFrameEnd()
            Manager._RunFrameEnd() #  swap buffers / time count / etc.

        EngineLogger.info('Engine loop Ended. Releasing Resources...')
        
        self.Stage = EngineStage.BEFORE_RELEASE
        self.beforeRelease()
        
        Manager._RunRelease()
        self.afterRelease()
        
        EngineLogger.success('Engine is ended.')
        self.Stage = EngineStage.ENDED
        
    @classmethod
    def Run(cls, *args, **kwargs):
        global _EngineInstance
        if _EngineInstance is None:
            _EngineInstance = cls(*args, **kwargs)
        _EngineInstance.run()