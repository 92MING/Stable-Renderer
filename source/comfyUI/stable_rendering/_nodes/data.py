from typing import Tuple, TYPE_CHECKING
from comfyUI.types import *
from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import is_dev_mode, is_verbose_mode
if TYPE_CHECKING:
    from common_utils.stable_render_utils import IDMap, SpriteInfos


class EngineDataNode(StableRenderingNode):
    '''
    Nodes providing the runtime data during rendering.
    `EngineData` is a hidden type, which means no input is required for this node,
    the data will be passed during runtime automatically.
    '''
    
    def __call__(self, engine_data: EngineData)->Tuple[
                        Named[IMAGE, "colors"],  # type: ignore
                        Named["IDMap", "ids"],     # type: ignore
                        Named[IMAGE, "positions"],    # type: ignore
                        Named[IMAGE, "normals"], # type: ignore
                        Named[IMAGE, "depths"],  # type: ignore
                        Named[LATENT, "noises"], # type: ignore
                        Named[MASK, "masks"],     # type: ignore
                        Named[CorrespondMaps, 'correspond_maps'],   # type: ignore
                        Named["SpriteInfos", "sprites"], # type: ignore
                        Named[EnvPrompts, "env_prompt"], # type: ignore
                ]:
        
        if engine_data is None:
                return (None, None, None, None, None, None, None, None, {}, "")
        
        return (engine_data.color_maps, 
                engine_data.id_maps, 
                engine_data.pos_maps, 
                engine_data.normal_maps,
                engine_data.depth_maps, 
                engine_data.noise_maps,
                engine_data.masks,
                engine_data.correspond_maps,
                engine_data.sprite_infos,
                engine_data.env_prompts)

    def IsChanged(self, engine_data: EngineData):
        from engine.engine import Engine
        if not Engine.IsLooping:
            return      # no need return anything for checking when engine is not running
        return Engine.Instance().RuntimeManager.FrameCount  # use frame count to determine if the frame data is changed


class InferenceOutputNode(StableRenderingNode):
    '''
    This is the node type defining the output data passing to `RenderManager` as rendering result.
    
    When u are running on the web UI, the `__server_call__` will be called instead of `__call__`,
    `InferenceOutput` will then act as `PreviewImage` to be displayed on the web UI.
    '''
    
    IsOutputNode = True
    '''this node is an output node(though it has no UI output).'''
    Unique = True
    '''only one instance of this node type is allowed in the graph.'''
    
    def __call__(self, 
                 colorImg: IMAGE, 
                 context: InferenceContext,      # this is hidden value, will be passed during runtime
                 )->InferenceOutput:
        out = InferenceOutput(colorImg)
        context.final_output = out
        return out
    
    # this will be called instead of `__call__` when web server is running
    def __server_call__(self,
                        colorImg: IMAGE,                # required type
                        context: InferenceContext,      # hidden type
                        prompt: PROMPT,                 # hidden type
                        png_info: EXTRA_PNG_INFO,       # hidden type
                        save: bool=False,               # optional type
                        ) -> UIImage:
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug('Calling InferenceOutputNode.__server_call__...')
        context.final_output = InferenceOutput(colorImg)
        return UIImage(colorImg, type='temp' if not save else 'output', prompt=prompt, png_info=png_info)
        
    
    
__all__ = ['EngineDataNode', 'InferenceOutputNode']