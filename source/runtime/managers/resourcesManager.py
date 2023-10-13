from .manager import Manager
from .sceneManager import SceneManager
from utils.global_utils import GetOrAddGlobalValue

class ResourcesManager(Manager):

    _PrepareFuncOrder = SceneManager._PrepareFuncOrder + 1  # after SceneManager (after all gameobj/components are created)
    _ReleaseFuncOrder = -1
    def _prepare(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()
        base_clses = [cls for cls in resources_clses if cls._BaseName == cls.ClsName()]
        for cls in sorted(base_clses, key=lambda cls: cls._LoadOrder):
            for instance in cls.AllInstances():
                try:
                    instance.sendToGPU()
                except Exception:
                    raise Exception(f'Error when sending {instance.__class__.__qualname__}:{instance.name} to GPU')

    def _release(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()
        base_clses = [cls for cls in resources_clses if cls._BaseName == cls.ClsName()]
        for cls in sorted(base_clses, key=lambda cls: cls._LoadOrder, reverse=True):
            for instance in cls.AllInstances():
                try:
                    instance.clear()
                except Exception:
                    raise Exception(f'Error when clearing {instance.__class__.__qualname__}:{instance.name}')