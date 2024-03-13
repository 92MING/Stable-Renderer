import OpenGL.GL as gl

from pathlib import Path
from typing import Union, Optional

from utils.decorators import Overload
from .ddsImage import *
from .texture import Texture


class Texture_DDS(Texture):
    '''Texture class for DDS images.'''

    _Format = 'dds'
    
    _dds: Optional[DDSImage] = None

    @Overload
    def load(self, path: Union[str, Path]):
        dds = DDSImage(path)
        self._dds = dds
        self._width = dds.width()
        self._height = dds.height()
        self._buffer = dds.data()

    def sendToGPU(self):
        if self._texID is not None:
            return
        
        self._texID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
        
        assert self._dds is not None, 'DDS image not loaded'
        format = self._dds.format()
        assert format is not None, 'Invalid DDS format'
        
        gl_format = format.getOpenGLformat()
        
        for level in range(0, self._dds.mip_count()):
            mipMap = self._dds.mip_map(level)
            width, height, offset, size = mipMap.width, mipMap.height, mipMap.offset, mipMap.size
            if self._dds.compressed():
                gl.glCompressedTexImage2D(gl.GL_TEXTURE_2D, level, gl_format, width, height, 0, self._buffer[offset:offset + size])
            else:
                gl.glTexImage2D(gl.GL_TEXTURE_2D, level, gl_format, width, height, 0, gl_format, gl.GL_UNSIGNED_BYTE, self._buffer[offset:offset + size])

    def clear(self):
        super().clear()
        self._dds = None


__all__ = ['Texture_DDS']