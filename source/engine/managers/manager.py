import traceback
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, List, Type
from common_utils.global_utils import GetOrCreateGlobalValue, GetGlobalValue, is_dev_mode
from common_utils.debug_utils import EngineLogger
from common_utils.type_utils import is_empty_method
if TYPE_CHECKING:
    from engine.engine import Engine

class ManagerFuncType(Enum):
    Prepare = 'prepare'
    Release = 'release'
    FrameBegin = 'begin'
    FrameRun = 'run'
    FrameEnd = 'end'

_MANAGERS = GetOrCreateGlobalValue('_ENGINE_MANAGERS', dict)
_MANAGER_FUNCS: dict[ManagerFuncType, List['Manager']] = GetOrCreateGlobalValue('_ENGINE_MANAGER_FUNCS', dict)
_MANAGER_CLSES = GetOrCreateGlobalValue('_ENGINE_MANAGER_CLSES', dict)

def _check_manager_has_debug_method(manager_cls: Type["Manager"], method_name:str):
    origin_method = getattr(Manager, method_name)
    sub_manager_method = getattr(manager_cls, method_name)
    if origin_method is sub_manager_method:
        return False
    if is_empty_method(sub_manager_method):
        return False
    return True

class _ManagerMeta(type):
    def __new__(cls, *args, **kwargs):
        cls_name = args[0]
        if cls_name in _MANAGER_CLSES:
            return _MANAGER_CLSES[cls_name]
        new_cls = super().__new__(cls, *args, **kwargs)
        if cls_name != 'Manager':
            _MANAGER_CLSES[cls_name] = new_cls
        return new_cls

class Manager(metaclass=_ManagerMeta):
    '''
    Manager are singletons classes that manage some specific functions in the engine during runtime.
    Public methods should be started with capital letter.
    '''

    _Engine: ClassVar['Engine'] = None  # type: ignore
    _Singleton: ClassVar['Manager'] = None  # type: ignore

    PrepareFuncOrder: ClassVar[int] = 0
    ReleaseFuncOrder: ClassVar[int] = 0
    FrameBeginFuncOrder: ClassVar[int] = 0
    FrameRunFuncOrder: ClassVar[int] = 0
    FrameEndFuncOrder: ClassVar[int] = 0
    
    _HasDebugOnFrameBeginMethod: ClassVar[bool] = False    
    _HasDebugOnFrameRunMethod: ClassVar[bool] = False
    _HasDebugOnFrameEndMethod: ClassVar[bool] = False
    _HasDebugPrepareMethod: ClassVar[bool] = False
    _HasDebugReleaseMethod: ClassVar[bool] = False

    def __init_subclass__(cls):
        cls._HasDebugOnFrameBeginMethod = _check_manager_has_debug_method(cls, 'debug_mode_on_frame_begin')
        cls._HasDebugOnFrameRunMethod = _check_manager_has_debug_method(cls, 'debug_mode_on_frame_run')
        cls._HasDebugOnFrameEndMethod = _check_manager_has_debug_method(cls, 'debug_mode_on_frame_end')
        cls._HasDebugPrepareMethod = _check_manager_has_debug_method(cls, 'debug_mode_prepare')
        cls._HasDebugReleaseMethod = _check_manager_has_debug_method(cls, 'debug_mode_release')

    def __new__(cls, *args, **kwargs):
        cls_name = cls.__qualname__
        if cls_name in _MANAGERS:
            manager_ins = _MANAGERS[cls_name]
            manager_ins.__init__ = lambda *a, **kw: None  # prevent re-initialization
            manager_ins.__class__.__init__ = lambda *a, **kw: None  # prevent re-initialization
        else:
            manager_ins = super().__new__(cls)
            cls._Singleton = manager_ins
            _MANAGERS[cls_name] = manager_ins

        return manager_ins

    def __init__(self):
        EngineLogger.print('Initializing manager: ', self.__class__.__qualname__)

    def on_frame_begin(self):
        '''The first function will be called in each frame.'''
        pass

    def debug_mode_on_frame_begin(self):
        '''In debug mode, this function will be called instead of "on_frame_begin".'''
        self.on_frame_begin() # override this function to debug

    def on_frame_run(self):
        '''Will be called after "on_frame_begin"'''
        pass

    def debug_mode_on_frame_run(self):
        '''In debug mode, this function will be called instead of "on_frame_run".'''
        self.on_frame_run()

    def on_frame_end(self):
        '''Will be called after "on_frame_run"'''
        pass

    def debug_mode_on_frame_end(self):
        '''In debug mode, this function will be called instead of "on_frame_end".'''
        self.on_frame_end()

    def prepare(self):
        '''prepare will be call before the loop begins.'''
        pass

    def debug_mode_prepare(self):
        '''In debug mode, this function will be called instead of "prepare".'''
        self.prepare()

    def release(self):
        '''release will be called when the program is going to exit.'''
        pass

    def debug_mode_release(self):
        '''In debug mode, this function will be called instead of "release".'''
        self.release()

    @property
    def engine(self)->'Engine':
        if not hasattr(self, '_engine') or self._Engine is None:
            self.__class__._Engine = GetGlobalValue('__ENGINE_INSTANCE__')    # type: ignore
        return self._Engine

    # region internal functions
    @staticmethod
    def _RunPrepare():
        if 'prepare' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.Prepare] = list(sorted(_MANAGERS.values(), key=lambda m: m.PrepareFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.Prepare]:
            try:
                manager.prepare() if not (manager.engine.IsDebugMode and manager._HasDebugPrepareMethod) else manager.debug_mode_prepare()
            except Exception as e:
                if is_dev_mode():
                    raise e
                EngineLogger.warn(f'(PREPARE) Error when running {manager.__class__.__qualname__} with msg: {e}, traceback:{traceback.format_exc()}')


    _last_run_frame_begin_msg = None
    _last_run_frame_run_msg = None
    _last_run_frame_end_msg = None
    @staticmethod
    def _RunFrameBegin():
        if 'begin' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameBegin] = list(sorted(_MANAGERS.values(), key=lambda m: m.FrameBeginFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameBegin]:
            try:
                manager.on_frame_begin() if not (manager.engine.IsDebugMode and manager._HasDebugOnFrameBeginMethod) else manager.debug_mode_on_frame_begin()
            except Exception as e:
                if is_dev_mode():
                    raise e
                trace = traceback.format_exc()
                msg = f'(FRAME BEGIN) Error when running {manager.__class__.__qualname__} with msg: {e}'
                if msg == manager._last_run_frame_begin_msg:
                    continue    # prevent duplicate message
                else:
                    manager._last_run_frame_begin_msg = msg
                    EngineLogger.warn(f'{msg}. Traceback:{trace}')

    @staticmethod
    def _RunFrameRun():
        if 'run' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameRun] =list(sorted(_MANAGERS.values(), key=lambda m: m.FrameRunFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameRun]:
            try:
                manager.on_frame_run() if not (manager.engine.IsDebugMode and manager._HasDebugOnFrameRunMethod) else manager.debug_mode_on_frame_run()
            except Exception as e:
                if is_dev_mode():
                    raise e
                trace = traceback.format_exc()
                msg = f'(FRAME RUN) Error when running {manager.__class__.__qualname__} with msg: {e}'
                if msg == manager._last_run_frame_run_msg:
                    continue
                else:
                    manager._last_run_frame_run_msg = msg
                    EngineLogger.warn(f'{msg}. Traceback:{trace}')

    @staticmethod
    def _RunFrameEnd():
        if 'end' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameEnd] = list(sorted(_MANAGERS.values(), key=lambda m: m.FrameEndFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameEnd]:
            try:
                manager.on_frame_end() if not (manager.engine.IsDebugMode and manager._HasDebugOnFrameEndMethod) else manager.debug_mode_on_frame_end()
            except Exception as e:
                if is_dev_mode():
                    raise e
                trace = traceback.format_exc()
                msg = f'(FRAME END) Error when running {manager.__class__.__qualname__} with msg: {e}'
                if msg == manager._last_run_frame_end_msg:
                    continue
                else:
                    manager._last_run_frame_end_msg = msg
                    EngineLogger.warn(f'{msg}. Traceback:{trace}')

    @staticmethod
    def _RunRelease():
        if 'release' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.Release] = list(sorted(_MANAGERS.values(), key=lambda m: m.ReleaseFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.Release]:
            try:
                manager.release() if not (manager.engine.IsDebugMode and manager._HasDebugReleaseMethod) else manager.debug_mode_release()
            except Exception as e:
                if is_dev_mode():
                    raise e
                EngineLogger.warn(f'(RELEASE) Error when running {manager.__class__.__qualname__} with msg: {e}, traceback:{traceback.format_exc()}')
    # endregion



__all__ = ['Manager']