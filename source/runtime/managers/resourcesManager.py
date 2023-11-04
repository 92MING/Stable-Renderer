import OpenGL.error
from .manager import Manager
from .sceneManager import SceneManager
from utils.global_utils import GetOrAddGlobalValue
import traceback

class ResourcesManager(Manager):

    _PrepareFuncOrder = SceneManager._PrepareFuncOrder + 1  # after SceneManager (after all gameobj/components are created)
    _ReleaseFuncOrder = 0 # release resources should be done at the beginning

    def _prepare(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()
        base_clses = [cls for cls in resources_clses if cls._BaseName == cls.ClsName()]
        for cls in sorted(base_clses, key=lambda cls: cls._LoadOrder):
            for instance in cls.AllInstances():
                try:
                    instance.sendToGPU()
                    if instance.__class__._BaseName == 'Texture':
                        print('Sent to GPU:', instance.__class__._BaseName + ':' + instance.name, ', texID:', instance.textureID)
                    else:
                        print('Sent to GPU:', instance.__class__._BaseName + ':' + instance.name)
                except Exception:
                    raise Exception(f'Error when sending {instance.__class__.__qualname__}:{instance.name} to GPU, traceback: {traceback.format_exc()}')

    def _release(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()
        base_clses = [cls for cls in resources_clses if cls._BaseName == cls.ClsName()]
        for cls in sorted(base_clses, key=lambda cls: cls._LoadOrder, reverse=True):
            for instance in cls.AllInstances():
                try:
                    instance.clear()
                    print('Cleared:', instance.__class__._BaseName + ':' + instance.name)
                except OpenGL.error.NullFunctionError:
                    pass # opengl already released, ignore
                except Exception:
                    raise Exception(f'Error when clearing {instance.__class__.__qualname__}:{instance.name}, traceback: {traceback.format_exc()}')