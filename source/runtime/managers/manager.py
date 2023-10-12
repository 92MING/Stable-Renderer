from utils.global_utils import GetOrAddGlobalValue, GetGlobalValue

_MANAGERS = GetOrAddGlobalValue('_ENGINE_MANAGERS', set())
_MANAGER_FUNCS = GetOrAddGlobalValue('_ENGINE_MANAGER_FUNCS', dict())
class Manager:

    _PrepareFuncOrder = 0
    _ReleaseFuncOrder = 0
    _FrameBeginFuncOrder = 0
    _FrameRunFuncOrder = 0
    _FrameEndFuncOrder = 0

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

    def _onFrameBegin(self):
        '''The first function will be called in each frame.'''
        pass
    def _onFrameRun(self):
        '''Will be called after "_onFrameBegin"'''
        pass
    def _onFrameEnd(self):
        '''Will be called after "_onFrameRun"'''
        pass
    def _prepare(self):
        '''Prepare will be call before the loop begins.'''
        pass
    def _release(self):
        '''Release will be called when the program is going to exit.'''
        pass

    @property
    def engine(self)->'Engine':
        return GetGlobalValue('_ENGINE_SINGLETON')

    @staticmethod
    def _RunPrepare():
        if 'prepare' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['prepare'] = sorted(_MANAGERS, key=lambda m: m._PrepareFuncOrder)
        for manager in _MANAGER_FUNCS['prepare']:
            manager._prepare()
    @staticmethod
    def _RunFrameBegin():
        if 'begin' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['begin'] = sorted(_MANAGERS, key=lambda m: m._FrameBeginFuncOrder)
        for manager in _MANAGER_FUNCS['begin']:
            manager._onFrameBegin()
    @staticmethod
    def _RunFrameRun():
        if 'run' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['run'] = sorted(_MANAGERS, key=lambda m: m._FrameRunFuncOrder)
        for manager in _MANAGER_FUNCS['run']:
            manager._onFrameRun()
    @staticmethod
    def _RunFrameEnd():
        if 'end' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['end'] = sorted(_MANAGERS, key=lambda m: m._FrameEndFuncOrder)
        for manager in _MANAGER_FUNCS['end']:
            manager._onFrameEnd()
    @staticmethod
    def _RunRelease():
        if 'release' not in _MANAGER_FUNCS:
            _MANAGER_FUNCS['release'] = sorted(_MANAGERS, key=lambda m: m._ReleaseFuncOrder)
        for manager in _MANAGERS:
            manager._release()

__all__ = ['Manager']