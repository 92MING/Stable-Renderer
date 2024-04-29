# *-* coding: utf-8 *-*
'''The base class for texture'''
import os
import sys
import OpenGL.GL as gl
import OpenGL.error
import numpy as np
import torch
import pycuda
import pycuda.gl
import pycuda.driver
import glm

from torch import Tensor
from pathlib import Path
from PIL import Image
from PIL.Image import Image as ImageType
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT, glInitTextureFilterAnisotropicEXT

from ..resourcesObj import ResourcesObj
from ..enums import *
from ..color import Color
from common_utils.decorators import Overload
from common_utils.debug_utils import EngineLogger
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue

from typing import TYPE_CHECKING, Union, Optional, Final, Literal

if TYPE_CHECKING:
    from ..shader import Shader


_SUPPORT_GL_CUDA_SHARE = GetOrAddGlobalValue('__SUPPORT_GL_CUDA_SHARE__', True)

_SUPPORT_ANISOTROPIC_FILTER = glInitTextureFilterAnisotropicEXT()
if _SUPPORT_ANISOTROPIC_FILTER:
    _MAX_ANISOTROPY = gl.glGetFloatv(GL_TEXTURE_MAX_ANISOTROPY_EXT)
else:
    _MAX_ANISOTROPY = None


class Texture(ResourcesObj):
    '''Base class for texture.'''

    _BaseName : Final['str'] = 'Texture'
    '''The base class name of all texture sub classes'''

    def __init__(self,
                 name: Optional[str],
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 format:TextureFormat=TextureFormat.RGB,
                 data: Optional[bytes]=None,
                 min_filter:TextureFilter=TextureFilter.LINEAR_MIPMAP_LINEAR,
                 mag_filter:TextureFilter=TextureFilter.LINEAR,
                 s_wrap:TextureWrap=TextureWrap.REPEAT,
                 t_wrap:TextureWrap=TextureWrap.REPEAT,
                 internalFormat: Optional[TextureInternalFormat]=None,
                 data_type: Optional[TextureDataType]=None,
                 share_to_torch: bool = False,
                 ):
        '''
        Texture datatype for this engine.
        
        Args:
            - name: The name of the texture.
            - width: The width of the texture
            - height: The height of the texture
            - format: The specific OpenGL format of the texture.
            - data: The data of the texture. It should be a bytes object.
            - min_filter: The minification filter of the texture.
            - mag_filter: The magnification filter of the texture.
            - s_wrap: The wrap mode of the texture in s direction.
            - t_wrap: The wrap mode of the texture in t direction.
            - internalFormat: The specific OpenGL internal format of the texture. If not given, it will be set to the default internal format of the format.
            - data_type: The specific OpenGL data type of the texture. If not given, it will be set to the default data type of the format.
            - share_to_torch: If true, the texture will be shared to torch by using cuda.
        '''
        super().__init__(name)
        self._texID = None
        '''The opengl texture id. It could be None if the texture is not loaded.'''
        self._cleared = False
        
        if width is not None and height is not None:
            assert width > 0 and height > 0, 'Invalid texture size, got width: {}, height: {}'.format(width, height)
        self._width = width
        self._height = height

        self._data = data
        '''Buffer for the texture data. It should be a bytes object.'''

        self._format = format
        self._min_filter = min_filter
        self._mag_filter = mag_filter
        self._s_wrap = s_wrap
        self._t_wrap = t_wrap
        self._internalFormat = internalFormat or self.format.default_internal_format
        self._data_type = data_type or self._internalFormat.value.default_data_type
        
        if share_to_torch and not self.data_type.value.tensor_supported:
            raise Exception(f'The data type {self.data_type} does not support tensor. Cannot share to torch.')
        
        self._share_to_torch = share_to_torch
        self._tensor: Optional[Tensor] = None
        self._support_gl_share_to_torch = True
    
    @property
    def textureID(self)->Optional[int]:
        '''The texture id for this texture in OpenGL. It could be None if the texture is not loaded.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self._texID
    
    @property
    def initedTensor(self)->bool:
        '''If true, the tensor sharing has been inited.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self._tensor is not None
    
    @property
    def sentToGPU(self)->bool:
        '''If true, the texture has been sent to GPU.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self._texID is not None
    
    @property
    def share_to_torch(self)->bool:
        '''If true, the texture will be shared to torch by using cuda.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self._share_to_torch

    @property
    def data(self)->Optional[bytes]:
        '''The data of the texture. It could be None if the texture is not loaded. It should be a bytes object.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self._data

    @property
    def width(self)->int:
        '''The width of the texture.'''
        assert not self._cleared, 'This texture has been cleared.'
        assert self._width is not None, 'Width is not set for this texture.'
        return self._width

    @property
    def height(self)->int:
        '''The height of the texture.'''
        assert not self._cleared, 'This texture has been cleared.'
        assert self._height is not None, 'Height is not set for this texture.'
        return self._height
    
    @property
    def nbytes(self)->int:
        '''The number of bytes of the texture.'''
        return self.width * self.height * self.format.channel_count
    
    @property
    def cell_nbytes(self)->int:
        '''return the number of bytes of a cell this texture.'''
        match self.data_type:
            case TextureDataType.UNSIGNED_BYTE:
                return self.format.channel_count * 1
            case TextureDataType.BYTE:
                return self.format.channel_count * 1
            
            case TextureDataType.UNSIGNED_SHORT:
                return self.format.channel_count * 2
            case TextureDataType.SHORT:
                return self.format.channel_count * 2
            
            case TextureDataType.UNSIGNED_INT:
                return self.format.channel_count * 4
            case TextureDataType.INT:
                return self.format.channel_count * 4
            
            case TextureDataType.FLOAT:
                return self.format.channel_count * 4
            case TextureDataType.HALF:
                return self.format.channel_count * 2
            
            case _:
                return self.data_type.value.nbytes
        
    @property
    def format(self)->TextureFormat:
        '''
        The specific OpenGL format of the texture. 
        See:
            * https://www.khronos.org/opengl/wiki/Image_Format
            * https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        '''
        assert not self._cleared, 'This texture has been cleared.'
        return self._format

    @format.setter
    def format(self, value: TextureFormat):
        '''
        The specific OpenGL format of the texture. 
        See:
            * https://www.khronos.org/opengl/wiki/Image_Format
            * https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        '''
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change tex format after texture is loaded.')
        assert isinstance(value, TextureFormat), 'Invalid format type: {}'.format(value)
        self._format = value
    
    @property
    def data_type(self) -> TextureDataType:
        '''
        The specific OpenGL data type of the texture.
        See:
            * https://www.khronos.org/opengl/wiki/Image_Format
            * https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        '''
        assert not self._cleared, 'This texture has been cleared.'
        return self._data_type
    
    @data_type.setter
    def data_type(self, value: TextureDataType):
        '''
        The specific OpenGL data type of the texture.
        See:
            * https://www.khronos.org/opengl/wiki/Image_Format
            * https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        '''
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change data type after texture is loaded.')
        #TODO: check if the datatype is valid for the current format
        self._data_type = value

    @property
    def internal_format(self)->TextureInternalFormat:
        assert not self._cleared, 'This texture has been cleared.'
        return self._internalFormat

    @internal_format.setter
    def internal_format(self, value:TextureInternalFormat):
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change internal format after texture is loaded.')
        # TODO: check if the internal format is valid for the current format
        self._internalFormat = value

    @property
    def channel_count(self)->int:
        '''The channel count of the texture.'''
        assert not self._cleared, 'This texture has been cleared.'
        return self.format.channel_count
    
    @property
    def min_filter(self) -> TextureFilter:
        assert not self._cleared, 'This texture has been cleared.'
        return self._min_filter

    @min_filter.setter
    def min_filter(self, value:TextureFilter):
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change filter after texture is loaded.')
        self._min_filter = value

    @property
    def mag_filter(self)->TextureFilter:
        assert not self._cleared, 'This texture has been cleared.'
        return self._mag_filter

    @mag_filter.setter
    def mag_filter(self, value:TextureFilter):
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change filter after texture is loaded.')
        self._mag_filter = value

    @property
    def s_wrap(self)->TextureWrap:
        assert not self._cleared, 'This texture has been cleared.'
        return self._s_wrap

    @s_wrap.setter
    def s_wrap(self, value):
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change s_wrap after texture is loaded.')
        self._s_wrap = value

    @property
    def t_wrap(self)->TextureWrap:
        assert not self._cleared, 'This texture has been cleared.'
        return self._t_wrap

    @t_wrap.setter
    def t_wrap(self, value: TextureWrap):
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is not None:
            raise Exception('Cannot change t_wrap after texture is loaded.')
        self._t_wrap = value
    # endregion


    @Overload
    def bind(self, slot:int, uniformID):    # type: ignore
        '''
        Bind texture to a slot and set shader uniform.
        Make sure you have used shader before calling this function.
        '''
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind.')
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)   # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
        gl.glUniform1i(uniformID, slot)
        
    @Overload
    def bind(self):
        '''call `glBindTexture` to bind the texture to the current context.'''
        assert not self._cleared, 'This texture has been cleared.'
        if self._texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind.')
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)

    @Overload
    def bind(self, slot:int, name:str, shader:'Shader'):
        '''Use shader, and bind texture to a slot and set shader uniform with your given name.'''
        assert not self._cleared, 'This texture has been cleared.'
        shader.useProgram()
        self.bind(slot, shader.getUniformID(name))


    @Overload
    def load(self, path: Union[str, Path]): # type: ignore
        '''Load the image on the given path by PIL.'''
        assert not self._cleared, 'This texture has been cleared.'
        EngineLogger.debug('Loading texture by path:' + str(path) + '...')
        image = Image.open(path).convert(self.format.PIL_convert_mode)
        self._data = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        self._height = image.height
        self._width = image.width
        image.close() 

    @Overload
    def load(self, array: np.ndarray):
        '''Load by numpy array'''
        assert not self._cleared, 'This texture has been cleared.'
        EngineLogger.debug('Loading texture by numpy array...')
        self._data = array.tobytes()
        self._width = array.shape[1]
        self._height = array.shape[0]
    
    @Overload
    def load(self, data: bytes, width: int, height: int):
        '''Load by bytes directly'''
        assert not self._cleared, 'This texture has been cleared.'
        EngineLogger.debug('Loading texture by bytes...')
        self._data = data
        self._width = width
        self._height = height

    @Overload
    def load(self, image: ImageType):
        '''Load by PIL image'''
        assert not self._cleared, 'This texture has been cleared.'
        EngineLogger.debug('Loading texture by PIL image...')
        self._data = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        self._height = image.height
        self._width = image.width
        image.close()
    
    @property
    def support_gl_share_to_torch(self):
        '''Whether the hardware support copying data from OpenGL to pytorch tensor directly.'''
        return self._support_gl_share_to_torch

    def _init_tensor(self):
        '''The tensor of the texture. It could be None if the texture is not shared to torch.'''
        assert not self._cleared, 'This texture has been cleared.'
        assert self._share_to_torch, 'This texture is not set to be sharing to torch.'
        assert self.textureID is not None, 'This texture is not yet sent to GPU.'
        
        if self._tensor is None:
            self._tensor = torch.zeros((self.height, self.width, self.channel_count), 
                                        dtype=self.data_type.value.torch_dtype, 
                                        device=f"cuda")
        if not GetOrAddGlobalValue('__SUPPORT_GL_CUDA_SHARE__', True):
            self._support_gl_share_to_torch = False
            return
        try:
            self._cuda_buffer = pycuda.gl.RegisteredImage(int(self.textureID),
                                                          int(gl.GL_TEXTURE_2D), 
                                                          pycuda.gl.graphics_map_flags.NONE)
        except Exception as e:
            SetGlobalValue('__SUPPORT_GL_CUDA_SHARE__', False)
            EngineLogger.warning(f'Warning: Texture {self.name} failed to register image. Error: {e}')
            self._support_gl_share_to_torch = False
        except:
            SetGlobalValue('__SUPPORT_GL_CUDA_SHARE__', False)
            EngineLogger.warning(f'Warning: Texture {self.name} failed to register image. Error: {sys.exc_info()[0]}')
            self._support_gl_share_to_torch = False
        
        if self.support_gl_share_to_torch:
            # prepare copiers. These copiers are used for copying data between GPU and GPU
            
            # for copying data from OpenGL to pytorch
            self._to_tensor_copier = pycuda.driver.Memcpy2D()  # type: ignore
            self._to_tensor_copier.width_in_bytes = self._to_tensor_copier.src_pitch = self._to_tensor_copier.dst_pitch = self.width * self.channel_count * self.data_type.value.nbytes
            self._to_tensor_copier.height = self.height
            self._to_tensor_copier.set_dst_device(self._tensor.data_ptr())
            
            # for copying data back from pytorch to OpenGL
            self._from_tensor_copier = pycuda.driver.Memcpy2D()  # type: ignore
            self._from_tensor_copier.width_in_bytes = self._from_tensor_copier.src_pitch = self._from_tensor_copier.dst_pitch = self.width * self.channel_count * self.data_type.value.nbytes
            self._from_tensor_copier.height = self.height
            self._from_tensor_copier.set_src_device(self._tensor.data_ptr())    # src device may change in `set_data` method
            
    def numpy_data(self, flipY: bool = True, level: int = 0)->np.ndarray:
        '''
        Copy the data of this texture from OpenGL, and return as a numpy array
        
        Args:
            - flipY: If true, the array will be flipped upside down. This is becuz the origin of OpenGL is at the bottom-left corner.
            - level: The mipmap level of the texture. Default is 0. This is only valid when the texture has mipmap.
        '''
        self.bind()
        data = gl.glGetTexImage(gl.GL_TEXTURE_2D, level, self.format.value.gl_format, self.data_type.value.gl_data_type)
        data = np.frombuffer(data, dtype=self.data_type.value.numpy_dtype)
        data = data.reshape((self.height, self.width, self.channel_count))
        if flipY:
            data = data[::-1, :, :]  # flip array upside down
        return data
    
    # from https://github.com/pytorch/pytorch/issues/107112
    def tensor(self, update:bool=True, flip: bool = True)->Tensor:
        '''
        Return the data of this texture in tensor.
        
        Note: If the hardware doesn't support register gl texture in cuda(so we cannot copy data from GPU to GPU directly), 
              this method will firstly get the data from GPU to CPU, and then copy it to GPU(which means it will be slow).
              
        Args:
            - update: If true, the tensor will be updated from GPU. If false, the tensor will be returned directly if it has been updated.
            - flip: If true, the tensor will be flipped upside down. This is becuz the origin of OpenGL is at the bottom-left corner.
        '''
        assert not self._cleared, 'This texture has been cleared.'
        assert self._texID, 'This texture is not yet sent to GPU.'
        
        if not update and self._tensor is not None:
            if flip:
                return self._tensor.flip(0)
            return self._tensor
        
        if self.share_to_torch and self.support_gl_share_to_torch:
            mapping = self._cuda_buffer.map()
            array = mapping.array(0, 0)
            self._to_tensor_copier.set_src_array(array)
            self._to_tensor_copier(aligned=False)  # this will update self._tensor by copying data from GPU to GPU
            torch.cuda.synchronize()
            mapping.unmap()
        else:
            numpy_data = self.numpy_data()
            self._tensor = torch.from_numpy(numpy_data).to(f"cuda:{self.engine.RenderManager.TargetDevice}")
        
        if flip:
            return self._tensor.flip(0) # type: ignore
        return self._tensor # type: ignore
    
    def sendToGPU(self):
        '''
        Create texture in GPU and send data.
        
        Note:
            This method could still be called even when self.data is None. 
            It means you are creating an empty texture(usually for FBO).
        '''
        assert not self._cleared, 'This texture has been cleared.'
        assert self._width is not None and self._height is not None, 'Invalid texture size, got width: {}, height: {}'.format(self._width, self._height)
        
        if self._texID is not None:
            return # already loaded

        self._texID = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
            
        data = self.data
        if not data:
            data = np.zeros((self.height, self.width, self.format.channel_count), 
                            dtype=self.data_type.value.numpy_dtype).tobytes()
        
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 
                        0, 
                        self.internal_format.value.gl_internal_format,
                        self.width, 
                        self.height, 
                        0, 
                        self.format.value.gl_format, 
                        self.data_type.value.gl_data_type,
                        data)  # data can be None, its ok
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.s_wrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.t_wrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter.value)
        
        if _SUPPORT_ANISOTROPIC_FILTER:
            gl.glTexParameterf(gl.GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, _MAX_ANISOTROPY)
            
        if (self.mag_filter in (TextureFilter.LINEAR_MIPMAP_LINEAR, TextureFilter.NEAREST_MIPMAP_NEAREST) or
            self.min_filter in (TextureFilter.LINEAR_MIPMAP_LINEAR, TextureFilter.NEAREST_MIPMAP_NEAREST)):
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        
        if self.share_to_torch:
            self._init_tensor()
            
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
                    EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(glErr))
            except Exception as err:
                EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(err))
            self._texID = None
        self._data = None
        self._width = None
        self._height = None
        self._tensor = None
        self._cleared = True
        del self

    def set_data(self, 
                 data: Union[bytes, Tensor, np.ndarray], 
                 xOffset: int = 0,
                 yOffset: int = 0,
                 width: Optional[int] = None,
                 height: Optional[int] = None):
        '''Write data to the texture. The data could be bytes, numpy array or torch tensor.'''
        
        width = width or (self.width - xOffset)
        height = height or (self.height - yOffset)
        self.bind()
        
        if isinstance(data, bytes):
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, self.format.value.gl_format, self.data_type.value.gl_data_type, data)
            
        elif isinstance(data, np.ndarray):
            if (data.shape[0] != height or data.shape[1] != width):
                if (data.shape[1] == height and data.shape[0] == width):
                    data = data.T
                else:
                    raise Exception('The data shape should be [height, width, channel_count]. Got: {}'.format(data.shape))
            
            if self.channel_count != data.shape[2]:
                if data.shape[2] < self.channel_count:
                    if data.shape[2] == 1:
                        data = np.repeat(data, repeats=self.channel_count, axis=2)
                    elif data.shape[2] == 3 and data.shape[0] == self.height and data.shape[1] == self.width:   # RGB
                        data = np.concatenate([data, np.ones_like(data[..., 0:1])], axis=-1)
                    else:
                        raise Exception('Invalid data shape: {}'.format(data.shape))
                else:
                    data = data[:, :, :self.channel_count]
            
            if data.dtype != self.data_type.value.numpy_dtype:
                data = data.astype(self.data_type.value.numpy_dtype)
            data = data.tobytes()
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, self.format.value.gl_format, self.data_type.value.gl_data_type, data)
            
        elif isinstance(data, Tensor):
            if (data.shape[0] != height or data.shape[1] != width):
                if (data.shape[1] == height and data.shape[0] == width):
                    data = data.T
                else:
                    raise Exception('The data shape should be [height, width, channel_count]. Got: {}'.format(data.shape))
            
            if self.channel_count != data.shape[2]:
                if data.shape[2] < self.channel_count:
                    if data.shape[2] == 1:
                        data = data.expand(-1, -1, self.channel_count)
                    elif data.shape[2] == 3 and data.shape[0] == self.height and data.shape[1] == self.width:   # RGB
                        data = torch.cat([data, torch.ones_like(data[..., 0:1])], dim=-1)
                    else:
                        raise Exception('Invalid data shape: {}'.format(data.shape))
                else:
                    data = data[:, :, :self.channel_count]
            
            if data.dtype != self.data_type.value.torch_dtype:
                data = data.to(self.data_type.value.torch_dtype)
            
            if self.share_to_torch and data.device.type == 'cuda':
                
                if data.device.index != self.engine.RenderManager.TargetDevice:
                    data = data.to(f"cuda:{self.engine.RenderManager.TargetDevice}")
                
                self._from_tensor_copier.height = height
                self._from_tensor_copier.width_in_bytes = self._from_tensor_copier.src_pitch = width * self.channel_count * self.data_type.value.nbytes
                self._from_tensor_copier.src_x = xOffset
                self._from_tensor_copier.src_y = yOffset
                self._from_tensor_copier.set_src_device(data.flatten().data_ptr())
                
                mapping = self._cuda_buffer.map()
                array = mapping.array(0, 0)
                self._from_tensor_copier.set_dst_array(array)
                self._from_tensor_copier(aligned=False) # copy data from GPU to GPU
                torch.cuda.synchronize()
                mapping.unmap()
                
            else:   # must copy data from CPU to GPU           
                data = data.cpu().numpy().tobytes()
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, self.format.value.gl_format, self.data_type.value.gl_data_type, data)
        else:
            raise Exception('Invalid data type: {}'.format(data))
        
    @classmethod
    def Load(cls,
             path: Union[str, Path], 
             name:Optional[str]=None,
             format:TextureFormat=TextureFormat.RGB,
             min_filter: TextureFilter = TextureFilter.LINEAR_MIPMAP_LINEAR,
             mag_filter: TextureFilter = TextureFilter.LINEAR,
             s_wrap:TextureWrap=TextureWrap.REPEAT,
             t_wrap:TextureWrap=TextureWrap.REPEAT,
             internalFormat: Optional[TextureInternalFormat]=None,
             data_type: Optional[TextureDataType]=None,
             share_to_torch: bool = False,
             )->'Texture':
        '''Create & load a texture from a file path. The format will be determined by the file extension.'''
        
        path, name = cls._GetPathAndName(path, name)

        if '.' in os.path.basename(path):
            ext = path.split('.')[-1]
            formatCls = cls.FindFormat(ext) # try to find a format class
            if formatCls is not None:
                texture = formatCls(name, 
                                    format=format, 
                                    min_filter=min_filter, 
                                    mag_filter =mag_filter, 
                                    s_wrap=s_wrap, 
                                    t_wrap=t_wrap,
                                    internalFormat=internalFormat,
                                    data_type=data_type,
                                    share_to_torch=share_to_torch)
                texture.load(path)
                return texture

        # default load by PIL
        texture = cls(name, 
                      format=format, 
                      min_filter=min_filter, 
                      mag_filter =mag_filter, 
                      s_wrap=s_wrap, 
                      t_wrap=t_wrap,
                      internalFormat=internalFormat,
                      data_type=data_type,
                      share_to_torch=share_to_torch)
        texture.load(path)
        return texture

    @staticmethod
    def CreateVirtualTex(name:Optional[str]=None,
                         height: int = 512,
                         width: int = 512,
                         fill: Union[Color, glm.vec3, glm.vec4, int, float] = Color.WHITE,
                         min_filter:TextureFilter=TextureFilter.LINEAR_MIPMAP_LINEAR,
                         mag_filter:TextureFilter=TextureFilter.LINEAR,
                         s_wrap:TextureWrap=TextureWrap.REPEAT,
                         t_wrap:TextureWrap=TextureWrap.REPEAT,
                         internalFormat: Optional[TextureInternalFormat] = None,
                         data_type: Optional[TextureDataType] = None,
                         share_to_torch: bool = False,
                         )->'Texture':
        '''Create an empty texture with a fill color as image.'''
        if not name:
            name = f"VirtualTex_{width}x{height}"
            count = 0
            while name in ResourcesObj.AllInstances():
                count += 1
                name = f"VirtualTex_{width}x{height}_{count}"
        
        if isinstance(fill, (int, float)):
            tex_format = TextureFormat.RED
            if not internalFormat and isinstance(fill, int):
                internalFormat = TextureInternalFormat.RED_16UI
        if isinstance(fill, glm.vec3):
            tex_format = TextureFormat.RGB
        elif isinstance(fill, (glm.vec4, Color)):
            tex_format = TextureFormat.RGBA
        else:
            raise Exception('Invalid fill type: {}'.format(fill))
        
        if isinstance(fill, (int, float)):
            fake_image = Image.new('L', (width, height), fill)
        elif isinstance(fill, glm.vec3):
            fake_image = Image.new('RGB', (width, height), (int(fill.x * 255), int(fill.y * 255), int(fill.z * 255)))
        elif isinstance(fill, glm.vec4):
            fake_image = Image.new('RGBA', (width, height), (int(fill.x * 255), int(fill.y * 255), int(fill.z * 255), int(fill.w * 255)))
        else:
            fake_image = Image.new('RGB', (width, height), (int(fill.r * 255), int(fill.g * 255), int(fill.b * 255)))
        
        texture = Texture(name=name,
                          width=width,
                          height=height,
                          format=tex_format,
                          data = fake_image.tobytes(),
                          min_filter=min_filter,
                          mag_filter=mag_filter,
                          s_wrap=s_wrap,
                          t_wrap=t_wrap,
                          internalFormat=internalFormat,
                          data_type=data_type,
                          share_to_torch=share_to_torch)
        
        return texture
    
    @staticmethod
    def CreateNoiseTex(name:Optional[str]=None,
                       height: Optional[int] = None,
                       width: Optional[int] = None,
                       sigma: float = 1.0,
                       channel_count = 4,
                       data_size: Literal[16, 32] = 16,
                       min_filter:TextureFilter=TextureFilter.LINEAR_MIPMAP_LINEAR,
                       mag_filter:TextureFilter=TextureFilter.LINEAR,
                       s_wrap:TextureWrap=TextureWrap.REPEAT,
                       t_wrap:TextureWrap=TextureWrap.REPEAT,
                       internalFormat: Optional[TextureInternalFormat] = None,
                       share_to_torch: bool = False,
                       )->'Texture':
        '''
        Create a texture which is filled with random gaussian noise in each pixel.
        
        Args:
            - name: The name of the texture.
            - height: The height of the texture.
            - width: The width of the texture.
            - sigma: The standard deviation of the gaussian noise.
            - channel_count: The channel count of the texture. It should be in [1, 4].
                             Default is 4, which is same as latent size.
            - data_size: The data size of the texture. It should be 16 or 32.
            - min_filter: The minification filter of the texture.
            - mag_filter: The magnification filter of the texture.
            - s_wrap: The wrap mode of the texture in s direction.
            - t_wrap: The wrap mode of the texture in t direction.
            - internalFormat: The specific OpenGL internal format of the texture. If not given, it will be set to the default internal format of the format.
            - share_to_torch: If true, the texture will be shared to torch by using cuda.
        '''
        
        from engine.engine import Engine
        height = height or Engine.Instance().WindowManager.WindowHeight
        width = width or Engine.Instance().WindowManager.WindowWidth
        
        if not name:
            name = f"RandomNoiseTex_{width}x{height}"
            count = 0
            while name in ResourcesObj.AllInstances():
                count += 1
                name = f"RandomNoiseTex_{width}x{height}_{count}"
        if channel_count < 1 or channel_count > 4:
            raise Exception(f'Invalid channel count: {channel_count}. It should be in [1, 4].')
        
        assert data_size in (16, 32), 'Invalid data size: {}'.format(data_size)
        dtype = {16: np.float16, 32: np.float32}[data_size]
        data = np.random.normal(0, sigma, (height, width, channel_count)).astype(dtype) # type: ignore
        
        if not internalFormat:
            if channel_count == 1:
                internalFormat = TextureInternalFormat.RED_32F if data_size == 32 else TextureInternalFormat.RED_16F
            elif channel_count == 2:
                internalFormat = TextureInternalFormat.RG32F if data_size == 32 else TextureInternalFormat.RG16F
            elif channel_count == 3:
                internalFormat = TextureInternalFormat.RGB32F if data_size == 32 else TextureInternalFormat.RGB16F
            elif channel_count == 4:
                internalFormat = TextureInternalFormat.RGBA32F if data_size == 32 else TextureInternalFormat.RGBA16F
            else:
                raise Exception('Invalid channel count: {}'.format(channel_count))
        
        tex = Texture(name=name,
                      width=width,
                      height=height,
                      format=TextureFormat.RGBA if channel_count == 4 else TextureFormat.RGB,
                      data=data.tobytes(),
                      min_filter=min_filter,
                      mag_filter=mag_filter,
                      s_wrap=s_wrap,
                      t_wrap=t_wrap,
                      internalFormat=internalFormat,
                      data_type=TextureDataType.FLOAT if data_size == 32 else TextureDataType.HALF,
                      share_to_torch=share_to_torch)
        return tex

    def __deepcopy__(self):
        '''
        For `custom_deep_copy` method in common_utils.type_utils.
        Cloning a texture will only clone the tensor(if it has). The texture will still point to the same OpenGL texture.
        '''
        tex = self.__class__(None,  # None means temporary obj
                            width=self.width,
                            height=self.height,
                            format=self.format,
                            data=self.data,
                            min_filter=self.min_filter,
                            mag_filter=self.mag_filter,
                            s_wrap=self.s_wrap,
                            t_wrap=self.t_wrap,
                            internalFormat=self.internal_format,
                            data_type=self.data_type,
                            share_to_torch=self.share_to_torch)
        tex._tensor = self._tensor.clone() if self._tensor is not None else None
        tex._texID = self._texID
        if self.share_to_torch and self.textureID is not None:  # when texID is not None, means sent to GPU
            tex._init_tensor()
        return tex
        
        
__all__ = ['Texture']