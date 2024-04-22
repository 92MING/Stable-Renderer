import OpenGL.error
from .manager import Manager
from .sceneManager import SceneManager
from common_utils.global_utils import GetOrAddGlobalValue
from common_utils.debug_utils import EngineLogger
import traceback

class ResourcesManager(Manager):

    PrepareFuncOrder = SceneManager.PrepareFuncOrder + 1  # after SceneManager (after all gameobj/components are created)
    ReleaseFuncOrder = 0 # release resources should be done at the beginning

    def prepare(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()

        prepared_instances = set()

        for cls in sorted(resources_clses, key=lambda cls: cls._LoadOrder):
            for instance in cls.AllInstances():
                if instance._internal_id in prepared_instances:
                    continue
                try:
                    instance.sendToGPU()
                    if instance.__class__._BaseName == 'Texture':
                        EngineLogger.print('Sent to GPU:', instance.__class__._BaseName + ':' + instance.name, ', texID:', instance.textureID)
                    else:
                        EngineLogger.print('Sent to GPU:', instance.__class__._BaseName + ':' + instance.name)
                except Exception:
                    raise Exception(f'Error when sending {instance.__class__.__qualname__}:{instance.name} to GPU, traceback: {traceback.format_exc()}')
                finally:
                    prepared_instances.add(instance._internal_id)

    def release(self):
        resources_clses = GetOrAddGlobalValue('_RESOURCES_CLSES', dict()).values()
        base_clses = [cls for cls in resources_clses if cls._BaseName == cls.ClsName()]
        released_instances = set()

        for cls in sorted(base_clses, key=lambda cls: cls._LoadOrder, reverse=True):
            for instance in cls.AllInstances():
                if instance._internal_id in released_instances:
                    continue
                try:
                    instance.clear()
                    EngineLogger.print('Cleared:', instance.__class__._BaseName + ':' + instance.name)

                except OpenGL.error.NullFunctionError:
                    pass # opengl already released, ignore

                except Exception:
                    raise Exception(f'Error when clearing {instance.__class__.__qualname__}:{instance.name}, traceback: {traceback.format_exc()}')

                finally:
                    released_instances.add(instance._internal_id)