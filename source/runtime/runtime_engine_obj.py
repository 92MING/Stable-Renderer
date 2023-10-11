from utils.global_utils import GetOrAddGlobalValue
from runtime.engine import Engine

class RuntimeEngineObj:
    @property
    def engine(self)->Engine:
        _engine = GetOrAddGlobalValue("_ENGINE_SINGLETON", None)
        if _engine is None:
            raise RuntimeError("Engine is not initialized yet.")
        return _engine