import cv2
import os.path
import numpy as np
import multiprocessing

from typing import TYPE_CHECKING, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pathlib import Path
from common_utils.debug_utils import EngineLogger
from common_utils.path_utils import *
from common_utils.global_utils import is_verbose_mode, is_dev_mode
from .manager import Manager
from engine.static.workflow import Workflow
from engine.static.enums import EngineMode

if TYPE_CHECKING:
    from comfyUI.types import FrameData
    from comfyUI.execution import PromptExecutor


class DiffusionManager(Manager):
    
    _prompt_executor: 'PromptExecutor'
    
    def __init__(self,
                 needOutputMaps=False,
                 outputCannyMap=True,
                 saveSDColorOutput=False,
                 maxFrameCacheCount=24,
                 mapSavingInterval=12,
                 threadPoolSize=None,
                 workflow: Union[Workflow, str, Path, None] = None):
        '''
        :param needOutputMaps: if output maps (id, color, depth...) result to disk
        :param maxFrameCacheCount: how many frames data should be stored in each
        :param mapSavingInterval: the frame interval between two map saving
        :param threadPoolSize: the size of thread pool. Threads are using for saving/loading maps asynchronously. If None, use the number of CPU cores.
        '''
        super().__init__()
        self._needOutputMaps = needOutputMaps
        self._maxFrameCacheCount = maxFrameCacheCount
        self._mapSavingInterval = mapSavingInterval
        self._saveSDColorOutput = saveSDColorOutput
        self._needOutputCannyMap = outputCannyMap
        
        if not threadPoolSize:
            threadPoolSize = multiprocessing.cpu_count()
        self._threadPool = ThreadPoolExecutor(max_workers=threadPoolSize) # for saving maps asynchronously
        self._outputPath = get_new_map_output_dir(create_if_not_exists=False)
        
        if not workflow:
            if self.engine.Mode == EngineMode.GAME:
                workflow = Workflow.DefaultGameWorkflow()
            elif self.engine.Mode == EngineMode.BAKE:
                workflow = Workflow.DefaultBakeWorkflow()
        else:
            if isinstance(workflow, (str, Path)):
                workflow = Workflow.Load(workflow)
        if is_dev_mode() and is_verbose_mode():
            EngineLogger.debug(f'Workflow is set to: {workflow}')
        self._workflow = workflow
        
    # region properties
    @property
    def NeedOutputCannyMap(self)->bool:
        return self._needOutputCannyMap
    
    @property
    def SaveSDColorOutput(self)->bool:
        return self._saveSDColorOutput
    
    @property
    def DefaultWorkflow(self)->Optional[Workflow]:
        '''the current running workflow'''
        return self._workflow
    
    @property
    def PromptExecutor(self)->"PromptExecutor":
        '''
        The PromptExecutor of ComfyUI. You can also get this in engine.
        Note that this can only be called when `
        '''
        return self.engine.PromptExecutor
    
    @property
    def ShouldOutputFrame(self):
        '''
        Return if the current frame's map data should be output to disk.
        This is different from `NeedOutputMaps`: `NeedOutputMaps` is a global flag, while `ShouldOutputFrame` will only be True when the current frame fullfills the condition set by `MapSavingInterval`.
        '''
        return self._needOutputMaps and self.engine.RuntimeManager.FrameCount % self._mapSavingInterval == 0
    
    @property
    def NeedOutputMaps(self)->bool:
        return self._needOutputMaps
    
    @NeedOutputMaps.setter
    def NeedOutputMaps(self, value:bool):
        self._needOutputMaps = value
        
    @property
    def MaxFrameCacheCount(self)->int:
        return self._maxFrameCacheCount
    
    @MaxFrameCacheCount.setter
    def MaxFrameCacheCount(self, value:int):
        self._maxFrameCacheCount = value
        
    @property
    def MapSavingInterval(self)->int:
        '''every how many frame should maps be saved(when the `NeedOutputMaps` is True)'''
        return self._mapSavingInterval
    
    @MapSavingInterval.setter
    def MapSavingInterval(self, value:int):
        self._mapSavingInterval = value
    # endregion

    # region outputs
    def _outputNumpyData(self, name:str, data:np.ndarray, frame_num: Optional[int]=None):
        '''output data in .npy format directly'''
        outputPath = os.path.join(self._outputPath, name)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        if frame_num is not None:
            np.save(os.path.join(outputPath, f'{name}_{frame_num}.npy'), data)
        else:
            np.save(os.path.join(outputPath, f'{name}.npy'), data)
        
    def OutputNumpyData(self, name:str, data:np.ndarray):
        '''
        output data in .npy format directly
        :param name: name of the map and folder. will be created if not exist. will be changed to lower case
        :param data: data to output
        :return:
        '''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputNumpyData, name, data, self.engine.RuntimeManager.FrameCount)
        
    def _outputMap(self, name:str, mapData:np.ndarray, multi255=True, dataType=np.uint8, frame_num:Optional[int]=None):
        '''this method will be pass to thread pool for saving maps asynchronously'''
        outputFormat = "RGBA"
        if multi255:
            mapData = mapData * 255
        
        if len(mapData.shape) == 2:
            mapData = np.expand_dims(mapData, axis=2)
            mapData = np.repeat(mapData, repeats=3, axis=2)
        elif mapData.shape[2] == 1:
            mapData = np.repeat(mapData, repeats=3, axis=2)
        
        if mapData.shape[2] == 3:   # add alpha channel
            alpha = np.ones((mapData.shape[0], mapData.shape[1], 1), dtype=dataType) * 255
            mapData = np.concatenate([mapData, alpha], axis=2)
        
        img = Image.fromarray(mapData.astype(dataType), outputFormat)
        name = name.lower()
        outputPath = os.path.join(self._outputPath, name)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        if frame_num is not None:
            img.save(os.path.join(outputPath, f'{name}_{frame_num}.png'))
        else:
            img.save(os.path.join(outputPath, f'{name}.png'))

    def OutputMap(self, name:str, mapData:np.ndarray, multi255=True, dataType=np.uint8):
        '''
        output map data to OUTPUT_DIR/runtime_map/name/year-month-day-hour_index.png
        :param name: name of the map and folder. will be created if not exist. will be changed to lower case
        :param mapData: map data to output
        :param multi255: if multiply 255 to map data
        :param dataType: data type of the map
        :return:
        '''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputMap, name, mapData, multi255, dataType, self.engine.RuntimeManager.FrameCount)

    def _outputDepthMap(self, mapData:np.ndarray, frame_num: Optional[int]=None):
        depth_data_max, depth_data_min = np.max(mapData), np.min(mapData[mapData > 0])
        diff = depth_data_max - depth_data_min
        if diff != 0:
            depth_data_normalized = (mapData - depth_data_min) / diff
        else:
            depth_data_normalized = mapData
        gray = (np.clip(depth_data_normalized, 0, 1) * 255).astype(np.uint8)
        alpha = (depth_data_normalized > 0).astype(np.uint8) * 255
        depth_img = Image.fromarray(np.stack([gray, gray, gray, alpha], axis=-1), 'RGBA')
        outputPath = os.path.join(self._outputPath, 'depth')
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        if frame_num is not None:
            depth_img.save(os.path.join(outputPath, f'depth_{frame_num}.png'))
        else:
            depth_img.save(os.path.join(outputPath, f'depth.png'))
            
    def OutputDepthMap(self, mapData:np.ndarray):
        '''output method especially for depth map'''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputDepthMap, mapData, self.engine.RuntimeManager.FrameCount)
    
    def _outputCannyMap(self, mapData:np.ndarray, frame_num: Optional[int]=None):
        mapData = (mapData * 255).astype(np.uint8)
        canny = cv2.Canny(mapData, 100, 200)
        outputPath = os.path.join(self._outputPath, 'canny')
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        if frame_num is not None:
            cv2.imwrite(os.path.join(outputPath, f'canny_{frame_num}.png'), canny)
        else:
            cv2.imwrite(os.path.join(outputPath, 'canny.png'), canny)
    
    def OutputCannyMap(self, mapData:np.ndarray):
        '''output method especially for canny map'''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputCannyMap, mapData, self.engine.RuntimeManager.FrameCount)
    # endregion

    # region diffusion
    def SubmitPrompt(self, frameData: "FrameData", workflow: Optional[Workflow]=None):
        '''submit prompt to comfyUI's prompt executor'''
        if not self.engine.PromptExecutor:
            if is_dev_mode():
                raise ValueError('No prompt executor is create(It maybe due to engine running with `startComfyUI=False`).')
            else:
                EngineLogger.error('No prompt executor is create(It maybe due to engine running with `startComfyUI=False`). Please ensure to run engine with comfyUI in production mode.')
                return None
        
        workflow = workflow or self.DefaultWorkflow
        if not workflow:
            if is_dev_mode():
                raise ValueError('No workflow is set. Please set a workflow before submitting prompt.')
            else:
                EngineLogger.error('No workflow is set. Please set a workflow before submitting prompt.')
                return None
        
        if not self.engine.IsLooping:
            EngineLogger.error('Engine is not looping. FrameData is not available. Cannot submit prompt to prompt executor.')
            return
        
        prompt, node_ids_to_be_ran, extra_data = workflow.build_prompt()
        context = self.engine.PromptExecutor.execute(prompt = prompt, 
                                                     extra_data = extra_data,
                                                     node_ids_to_be_ran=node_ids_to_be_ran, 
                                                     frame_data=frameData)
        if self.SaveSDColorOutput:
            final_output = context.final_output
            if final_output is None:
                EngineLogger.error('(DiffusionManager) No final output is found in context. Cannot save color output.')
            else:
                saving_dir = get_sd_color_result_dir()
                os.makedirs(saving_dir, exist_ok=True)
                saving_path = os.path.join(saving_dir, f'{self.engine.RuntimeManager.FrameCount}.png')
                color_img = final_output.frame_color
                def save(img):
                    img = 255. * color_img[0].cpu().numpy()
                    img_bytes = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                    img_bytes.save(saving_path)
                    if is_dev_mode():
                        EngineLogger.info(f'(DiffusionManager) Color output saved to {saving_path}')
                self._threadPool.submit(save, color_img)
                
        return context
    # endregion
    

__all__ = ['DiffusionManager']