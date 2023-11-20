from utils.global_utils import GetGlobalValue

class _classproperty:
    def __init__(self, func):
        self._func = func
    def __get__(self, cls, owner):
        return self._func(owner)

class EngineObj:
    _engine = None
    @_classproperty
    def engine(cls)->'Engine':
        if cls._engine is None:
            engine = GetGlobalValue("_ENGINE_SINGLETON")
            if engine is None:
                raise RuntimeError("Engine is not initialized yet.")
            cls._engine = engine
        return cls._engine