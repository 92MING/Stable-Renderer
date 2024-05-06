import traceback
from .manager import Manager
from .sceneManager import SceneManager
from common_utils.global_utils import GetOrCreateGlobalValue
from common_utils.debug_utils import EngineLogger
from typing import List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from engine.static.resources_obj import ResourcesObj


class ResourcesManager(Manager):

    PrepareFuncOrder = SceneManager.PrepareFuncOrder + 1  # after SceneManager (after all gameobj/components are created)
    ReleaseFuncOrder = 0 # release resources should be done at the beginning

    def prepare(self):
        prepared_instances = set()
        from engine.static.resources_obj import __TO_BE_LOAD_RESOURCES__
        
        while len(__TO_BE_LOAD_RESOURCES__) > 0:
            to_be_load: List['ResourcesObj'] = list(__TO_BE_LOAD_RESOURCES__)
            __TO_BE_LOAD_RESOURCES__.clear()
            to_be_load.sort(key=lambda obj: obj.LoadOrder)
            
            for instance in to_be_load:
                if instance.id in prepared_instances:
                    continue
                elif instance.loaded:
                    prepared_instances.add(instance.id)
                    continue
                
                try:
                    instance.load()
                except Exception:
                    raise Exception(f'Error when sending {instance} to GPU, traceback: {traceback.format_exc()}')
                finally:
                    prepared_instances.add(instance.id)

    def release(self):
        released_instances = set()
        from engine.static.resources_obj import __TO_BE_DESTROY_RESOURCES__
        
        while len(__TO_BE_DESTROY_RESOURCES__) > 0:
            to_be_destroy = list(__TO_BE_DESTROY_RESOURCES__)
            __TO_BE_DESTROY_RESOURCES__.clear()
            
            for instance in to_be_destroy:
                if instance.id in released_instances:
                    continue
                elif not instance.loaded or instance._destroyed:
                    released_instances.add(instance.id)
                    continue

                try:
                    instance.destroy()
                except Exception:
                    raise Exception(f'Error when destroying {instance}, traceback: {traceback.format_exc()}')
                finally:
                    released_instances.add(instance.id)