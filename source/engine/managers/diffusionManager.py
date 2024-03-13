import os.path
import OpenGL.GL as gl
import numpy as np
import glm
import multiprocessing

from .manager import Manager
from utils.path_utils import *
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from PIL import Image


@dataclass
class FrameData:
    '''
    Map data of 1 single frame.
    Note that img data is upside down since OpenGL is bottom-left origin while OpenCV is top-left origin.
    '''
    frameIndex: int
    idData: np.ndarray
    colorData: np.ndarray
    posData: np.ndarray
    normalData: np.ndarray
    depthData: np.ndarray

    def getPixelData(self, x, y):
        y = self.colorData.shape[0] - y - 1
        return {
            'id': self.idData[y, x], # [objID, xCoord, yCoord]
            'color': self.colorData[y, x], # [r, g, b]
            'pos': self.posData[y, x], # [x, y, z]
            'normal': self.normalData[y, x], # [x, y, z]
            'depth': self.depthData[y, x], # [z]
        }


class DiffusionManager(Manager):
    def __init__(self,
                 needOutputMaps=False,
                 mapMinimizeRatio=64,
                 maxFrameCacheCount=24,
                 mapSavingInterval=12,
                 threadPoolSize=None,):
        '''
        :param needOutputMaps: if output maps (id, color, depth...) result to disk
        :param maxFrameCacheCount: how many frames data should be stored in each
        :param mapSavingInterval: the frame interval between two map saving
        :param threadPoolSize: the size of thread pool. Threads are using for saving/loading maps asynchronously. If None, use the number of CPU cores.
        '''
        super().__init__()
        self._needOutputMaps = needOutputMaps
        self._mapMinimizeRatio = mapMinimizeRatio
        self._maxFrameCacheCount = maxFrameCacheCount
        self._mapSavingInterval = mapSavingInterval

        if not threadPoolSize:
            threadPoolSize = multiprocessing.cpu_count()
        self._threadPool = ThreadPoolExecutor(max_workers=threadPoolSize) # for saving maps asynchronously
        self._outputPath = get_new_map_output_dir(create_if_not_exists=False)

    # region properties
    @property
    def MapMinimizeRatio(self)->int:
        return self._mapMinimizeRatio
    
    @MapMinimizeRatio.setter
    def MapMinimizeRatio(self, value:int):
        '''Ratio should be a perfect square'''
        assert (value > 0 and glm.sqrt(value) == int(glm.sqrt(value))), 'MapMinimizeRatio must be a perfect square'
        self._mapMinimizeRatio = value
        
    @property
    def ShouldOutputFrame(self):
        '''Return if the current frame's map data should be output to disk'''
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
        return self._mapSavingInterval
    
    @MapSavingInterval.setter
    def MapSavingInterval(self, value:int):
        self._mapSavingInterval = value
    # endregion

    # region output maps
    def _outputNumpyData(self, name:str, data:np.ndarray):
        '''output data in .npy format directly'''
        outputPath = os.path.join(self._outputPath, name)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        np.save(os.path.join(outputPath, f'{name}_{self.engine.RuntimeManager.FrameCount}.npy'), data)
        
    def OutputNumpyData(self, name:str, data:np.ndarray):
        '''
        output data in .npy format directly
        :param name: name of the map and folder. will be created if not exist. will be changed to lower case
        :param data: data to output
        :return:
        '''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputNumpyData, name, data)
        
    def _outputMap(self, name:str, mapData:np.ndarray, multi255=True, dataType=np.uint8):
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
        img.save(os.path.join(outputPath, f'{name}_{self.engine.RuntimeManager.FrameCount}.png'))

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
            self._threadPool.submit(self._outputMap, name, mapData, multi255, dataType)

    def _outputDepthMap(self, mapData:np.ndarray):
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
        depth_img.save(os.path.join(outputPath, f'depth_{self.engine.RuntimeManager.FrameCount}.png'))

    def OutputDepthMap(self, mapData:np.ndarray):
        '''output method especially for depth map'''
        if self._needOutputMaps:
            self._threadPool.submit(self._outputDepthMap, mapData)
    # endregion



__all__ = ['DiffusionManager']