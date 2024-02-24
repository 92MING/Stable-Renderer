import traceback
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, List
from utils.global_utils import GetOrCreateGlobalValue, GetGlobalValue
if TYPE_CHECKING:
    from ..engine import Engine

class ManagerFuncType(Enum):
    Prepare = 'prepare'
    Release = 'release'
    FrameBegin = 'begin'
    FrameRun = 'run'
    FrameEnd = 'end'

_MANAGERS = GetOrCreateGlobalValue('_ENGINE_MANAGERS', dict)
_MANAGER_FUNCS: dict[ManagerFuncType, List['Manager']] = GetOrCreateGlobalValue('_ENGINE_MANAGER_FUNCS', dict)
_MANAGER_CLSES = GetOrCreateGlobalValue('_ENGINE_MANAGER_CLSES', dict)

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

    _engine: ClassVar['Engine'] = None
    _singleton: ClassVar['Manager'] = None

    PrepareFuncOrder: ClassVar[int] = 0
    ReleaseFuncOrder: ClassVar[int] = 0
    FrameBeginFuncOrder: ClassVar[int] = 0
    FrameRunFuncOrder: ClassVar[int] = 0
    FrameEndFuncOrder: ClassVar[int] = 0

    def __new__(cls, *args, **kwargs):
        cls_name = cls.__qualname__
        if cls_name in _MANAGERS:
            manager_ins = _MANAGERS[cls_name]
            manager_ins.__init__ = lambda *a, **kw: None  # prevent re-initialization
            manager_ins.__class__.__init__ = lambda *a, **kw: None  # prevent re-initialization
        else:
            manager_ins = super().__new__(cls)
            cls._singleton = manager_ins
            _MANAGERS[cls_name] = manager_ins

        return manager_ins

    def __init__(self):
        print('Initializing manager: ', self.__class__.__qualname__)

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
        if not hasattr(self, '_engine') or self._engine is None:
            self.__class__._engine = GetGlobalValue('_ENGINE_SINGLETON')
        return self._engine

    # region internal functions
    @staticmethod
    def _RunPrepare():
        if 'prepare' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.Prepare] = list(sorted(_MANAGERS.values(), key=lambda m: m.PrepareFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.Prepare]:
            try:
                manager.prepare() if not manager.engine.IsDebugMode else manager.debug_mode_prepare()
            except Exception as e:
                print(f'Warning: Error when running "prepare" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameBegin():
        if 'begin' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameBegin] = list(sorted(_MANAGERS.values(), key=lambda m: m.FrameBeginFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameBegin]:
            try:
                manager.on_frame_begin() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_begin()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_begin" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameRun():
        if 'run' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameRun] =list(sorted(_MANAGERS.values(), key=lambda m: m.FrameRunFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameRun]:
            try:
                manager.on_frame_run() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_run()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_run" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameEnd():
        if 'end' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.FrameEnd] = list(sorted(_MANAGERS.values(), key=lambda m: m.FrameEndFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.FrameEnd]:
            try:
                manager.on_frame_end() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_end()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_end" of {manager.__class__.__qualname__}.Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunRelease():
        if 'release' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS[ManagerFuncType.Release] = list(sorted(_MANAGERS.values(), key=lambda m: m.ReleaseFuncOrder))
        for manager in _MANAGER_FUNCS[ManagerFuncType.Release]:
            try:
                manager.release() if not manager.engine.IsDebugMode else manager.debug_mode_release()
            except Exception as e:
                print(f'Warning: Error when running "release" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')
    # endregion



__all__ = ['Manager']