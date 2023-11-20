import OpenGL.error

from static.resourcesObj import ResourcesObj
import os
from PIL import Image
from static.enums import *
from utils.decorator import Overload
import numpy as np

class Texture(ResourcesObj):
    _BaseName = 'Texture'

    def __init__(self, name,
                 format:TextureFormat=TextureFormat.RGB,
                 min_filter:TextureFilter=TextureFilter.LINEAR_MIPMAP_LINEAR,
                 mag_filter:TextureFilter=TextureFilter.LINEAR,
                 s_wrap:TextureWrap=TextureWrap.REPEAT,
                 t_wrap:TextureWrap=TextureWrap.REPEAT,
                 internalFormat=None # you could specify a different internal format
                 ):
        super().__init__(name)
        self._texID = None
        self._format = format
        self._min_filter = min_filter
        self._mag_filter = mag_filter
        self._s_wrap = s_wrap
        self._t_wrap = t_wrap
        self._width = None
        self._height = None
        self._buffer = None
        self._internalFormat = internalFormat

    # region properties
    @property
    def textureID(self):
        return self._texID
    @property
    def internalFormat(self):
        return self._internalFormat or self.format.get_default_internal_format()
    @property
    def buffer(self):
        return self._buffer
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def format(self):
        return self._format
    @format.setter
    def format(self, value):
        if self._texID is not None:
            raise Exception('Cannot change tex format after texture is loaded.')
        self._format = value
    @property
    def min_filter(self):
        return self._min_filter
    @min_filter.setter
    def min_filter(self, value:TextureFilter):
        if self._texID is not None:
            raise Exception('Cannot change filter after texture is loaded.')
        self._min_filter = value
    @property
    def mag_filter(self):
        return self._mag_filter
    @mag_filter.setter
    def mag_filter(self, value:TextureFilter):
        if self._texID is not None:
            raise Exception('Cannot change filter after texture is loaded.')
        self._mag_filter = value
    @property
    def s_wrap(self):
        return self._s_wrap
    @s_wrap.setter
    def s_wrap(self, value):
        if self._texID is not None:
            raise Exception('Cannot change s_wrap after texture is loaded.')
        self._s_wrap = value
    @property
    def t_wrap(self):
        return self._t_wrap
    @t_wrap.setter
    def t_wrap(self, value):
        if self._texID is not None:
            raise Exception('Cannot change t_wrap after texture is loaded.')
        self._t_wrap = value
    # endregion

    @Overload
    def bind(self, slot:int, uniformID):
        '''
        Bind texture to a slot and set shader uniform.
        Make sure you have used shader before calling this function.
        '''
        if self._texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind.')
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
        gl.glUniform1i(uniformID, slot)
    @Overload
    def bind(self, slot:int, name:str, shader:'Shader'):
        '''Use shader, and bind texture to a slot and set shader uniform with your given name.'''
        shader.useProgram()
        self.bind(slot, shader.getUniformID(name))

    def load(self, path):
        '''Default load by PIL'''
        image = Image.open(path).convert(self.format.get_PIL_convert_mode())
        self._buffer = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        self._height = image.height
        self._width = image.width
        image.close()
    def sendToGPU(self):
        '''Send data to GPU memory'''
        if self._texID is not None:
            return # already loaded
        self._texID = gl.glGenTextures(1)

        if self.format == TextureFormat.DEPTH_COMPONENT:
            type = gl.GL_UNSIGNED_INT
        elif self.format == TextureFormat.DEPTH_STENCIL:
            type = gl.GL_UNSIGNED_INT_24_8
        else:
            type = gl.GL_UNSIGNED_BYTE

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internalFormat, self.width, self.height, 0, self.format.value, type, self.buffer)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.s_wrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.t_wrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter.value)
        if (self.mag_filter in (TextureFilter.LINEAR_MIPMAP_LINEAR, TextureFilter.NEAREST_MIPMAP_NEAREST) or
            self.min_filter in (TextureFilter.LINEAR_MIPMAP_LINEAR, TextureFilter.NEAREST_MIPMAP_NEAREST)):
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    def clear(self):
        '''
        Clear all data and release GPU memory if it has been loaded.
        This is a default implementation, you may need to override it.
        '''
        super().clear()
        if self._texID is not None:
            try:
                buffer = np.array([self._texID], dtype=np.uint32)
                gl.glDeleteTextures(1, buffer)
            except OpenGL.error.GLError as glErr:
                if glErr.err == 1282:
                    pass
                else:
                    print('Warning: failed to delete texture. Error: {}'.format(glErr))
            except Exception as err:
                print('Warning: failed to delete texture. Error: {}'.format(err))
            self._texID = None
        self._buffer = None
        self._width = None
        self._height = None
        self._internalFormat = None

    @classmethod
    def Load(cls, path, name=None,
             format:TextureFormat=TextureFormat.RGB,
             min_filter: TextureFilter = TextureFilter.LINEAR_MIPMAP_LINEAR,
             mag_filter: TextureFilter = TextureFilter.LINEAR,
             s_wrap:TextureWrap=TextureWrap.REPEAT,
             t_wrap:TextureWrap=TextureWrap.REPEAT)->'Texture':

        path, name = cls._GetPathAndName(path, name)

        if '.' in os.path.basename(path):
            ext = path.split('.')[-1]
            formatCls = cls.FindFormat(ext) # try to find a format class
            if formatCls is not None:
                texture = formatCls(name, format, min_filter, mag_filter, s_wrap, t_wrap)
                texture.load(path)
                return texture

        # default load by PIL
        texture = cls(name, format, min_filter, mag_filter, s_wrap, t_wrap)
        texture.load(path)
        return texture

__all__ = ['Texture']