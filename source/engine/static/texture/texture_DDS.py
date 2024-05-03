import OpenGL.GL as gl

from attr import attrib, attrs
from pathlib import Path
from typing import Union, Optional

from common_utils.global_utils import is_dev_mode
from engine.static.enums import TextureFormat, TextureFilter, TextureWrap, TextureInternalFormat, TextureDataType
from .ddsImage import *
from .texture import Texture

@attrs(eq=False, repr=False)
class Texture_DDS(Texture):
    '''Texture class for DDS images.'''

    Format = 'dds'
    
    dds: Optional[DDSImage] = attrib(default=None, kw_only=True)

    @classmethod
    def Load(cls, 
             path: Union[str, Path],
             name:Optional[str]=None,
             format:TextureFormat=TextureFormat.RGB,
             min_filter: TextureFilter = TextureFilter.LINEAR_MIPMAP_LINEAR,
             mag_filter: TextureFilter = TextureFilter.LINEAR,
             s_wrap:TextureWrap=TextureWrap.REPEAT,
             t_wrap:TextureWrap=TextureWrap.REPEAT,
             internal_format: Optional[TextureInternalFormat]=None,
             data_type: Optional[TextureDataType]=None,
             share_to_torch: bool = False,):
        '''Load DDS image from path.'''
        dds = DDSImage(path)
        width = dds.width()
        height = dds.height()
        buffer = dds.data()
        internal_format = internal_format or format.default_internal_format
        data_type = data_type or internal_format.value.default_data_type
        
        tex = cls(
            name=name,
            dds=dds,
            format=format,
            width=width,
            height=height,
            data=buffer,
            min_filter=min_filter,
            mag_filter=mag_filter,
            s_wrap=s_wrap,
            t_wrap=t_wrap,
            internal_format=internal_format,
            data_type=data_type,
            share_to_torch=share_to_torch,
        )
        return tex
        

    def load(self):
        if self.loaded:
            return        
        if self.data is None:
            if is_dev_mode():
                raise ValueError('No data loaded')
            else:
                return
        super().load()
        
        self.texID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texID)
        
        assert self.dds is not None, 'DDS image not loaded'
        format = self.dds.format()
        assert format is not None, 'Invalid DDS format'
        
        gl_format = format.getOpenGLformat()
        
        for level in range(0, self.dds.mip_count()):
            mipMap = self.dds.mip_map(level)
            width, height, offset, size = mipMap.width, mipMap.height, mipMap.offset, mipMap.size
            if self.dds.compressed():
                gl.glCompressedTexImage2D(gl.GL_TEXTURE_2D, level, gl_format, width, height, 0, self.data[offset:offset + size])
            else:
                gl.glTexImage2D(gl.GL_TEXTURE_2D, level, gl_format, width, height, 0, gl_format, gl.GL_UNSIGNED_BYTE, self.data[offset:offset + size])

    def clear(self):
        super().clear()
        self.dds = None



__all__ = ['Texture_DDS']