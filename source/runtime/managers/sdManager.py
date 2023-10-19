from .manager import Manager
from utils.path_utils import *
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np

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

class SDManager(Manager):
    def __init__(self,
                 outputMaps=False,
                 maxFrameCacheCount=24,
                 mapSavingInterval=12,
                 threadPoolSize=6,):
        '''
        :param outputMaps: if output maps result to disk
        :param maxFrameCacheCount: how many frames data should be stored in each
        :param mapSavingInterval: the frame interval between two map saving
        :param threadPoolSize: the size of thread pool. Threads are using for saving/loading maps asynchronously
        '''
        super().__init__()
        self._outputMaps = outputMaps
        self._maxFrameCacheCount = maxFrameCacheCount
        self._mapSavingInterval = mapSavingInterval

        self._threadPool = ThreadPoolExecutor(max_workers=threadPoolSize)

    # region properties
    @property
    def OutputMaps(self)->bool:
        return self._outputMaps
    @OutputMaps.setter
    def OutputMaps(self, value:bool):
        self._outputMaps = value
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



__all__ = ['SDManager']