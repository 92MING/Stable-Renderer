from utils.global_utils import GetOrAddGlobalValue, GetGlobalValue
import traceback
from typing import TYPE_CHECKING, ClassVar
if TYPE_CHECKING:
    from runtime.engine import Engine

_MANAGERS = GetOrAddGlobalValue('_ENGINE_MANAGERS', set())
_MANAGER_FUNCS = GetOrAddGlobalValue('_ENGINE_MANAGER_FUNCS', dict())

class Manager:
    '''
    Manager are singletons classes that manage some specific functions in the engine during runtime.
    Public methods should be started with capital letter.
    '''

    _engine: 'Engine' = None

    PrepareFuncOrder: ClassVar[int] = 0
    ReleaseFuncOrder: ClassVar[int] = 0
    FrameBeginFuncOrder: ClassVar[int] = 0
    FrameRunFuncOrder: ClassVar[int] = 0
    FrameEndFuncOrder: ClassVar[int] = 0

    def __new__(cls, *args, **kwargs):
        for manager in _MANAGERS:
            if manager.__class__.__qualname__ == cls.__qualname__:
                manager.__init__ = lambda *a, **kw: None
                return manager
        obj = super().__new__(cls)
        _MANAGERS.add(obj)
        return obj

    def __init__(self):
        print('Initializing manager: ', self.__class__.__qualname__)

    def on_frame_begin(self):
        '''The first function will be called in each frame.'''
        pass

    def on_frame_run(self):
        '''Will be called after "on_frame_begin"'''
        pass

    def on_frame_end(self):
        '''Will be called after "on_frame_run"'''
        pass

    def debug_mode_on_frame_begin(self):
        '''In debug mode, this function will be called instead of "on_frame_begin".'''
        self.on_frame_begin() # override this function to debug

    def debug_mode_on_frame_run(self):
        '''In debug mode, this function will be called instead of "on_frame_run".'''
        self.on_frame_run()

    def debug_mode_on_frame_end(self):
        '''In debug mode, this function will be called instead of "on_frame_end".'''
        self.on_frame_end()

    def prepare(self):
        '''prepare will be call before the loop begins.'''
        pass

    def release(self):
        '''release will be called when the program is going to exit.'''
        pass

    @property
    def engine(self)->'Engine':
        if self._engine is None:
            self._engine = GetGlobalValue('_ENGINE_SINGLETON')
        return self._engine

    # region internal functions
    @staticmethod
    def _RunPrepare():
        if 'prepare' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['prepare'] = sorted(_MANAGERS, key=lambda m: m.PrepareFuncOrder)
        for manager in _MANAGER_FUNCS['prepare']:
            try:
                manager.prepare()
            except Exception as e:
                print(f'Warning: Error when running "prepare" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameBegin():
        if 'begin' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['begin'] = sorted(_MANAGERS, key=lambda m: m.FrameBeginFuncOrder)
        for manager in _MANAGER_FUNCS['begin']:
            try:
                manager.on_frame_begin() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_begin()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_begin" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameRun():
        if 'run' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['run'] = sorted(_MANAGERS, key=lambda m: m.FrameRunFuncOrder)
        for manager in _MANAGER_FUNCS['run']:
            try:
                manager.on_frame_run() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_run()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_run" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunFrameEnd():
        if 'end' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['end'] = sorted(_MANAGERS, key=lambda m: m.FrameEndFuncOrder)
        for manager in _MANAGER_FUNCS['end']:
            try:
                manager.on_frame_end() if not manager.engine.IsDebugMode else manager.debug_mode_on_frame_end()
            except Exception as e:
                print(f'Warning: Error when running "on_frame_end" of {manager.__class__.__qualname__}.Err msg: {e}, traceback:{traceback.format_exc()}')

    @staticmethod
    def _RunRelease():
        if 'release' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['release'] = sorted(_MANAGERS, key=lambda m: m.ReleaseFuncOrder)
        for manager in _MANAGERS:
            try:
                manager.release()
            except Exception as e:
                print(f'Warning: Error when running "release" of {manager.__class__.__qualname__}. Err msg: {e}, traceback:{traceback.format_exc()}')
    # endregion



__all__ = ['Manager']