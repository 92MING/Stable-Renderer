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

from attr import attrs, attrib
from torch import Tensor
from pathlib import Path
from PIL import Image
from PIL.Image import Image as ImageType
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT, glInitTextureFilterAnisotropicEXT

from ..resources_obj import ResourcesObj
from ..enums import *
from ..color import Color
from common_utils.decorators import Overload
from common_utils.debug_utils import EngineLogger
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue, is_dev_mode

from typing import TYPE_CHECKING, Union, Optional, Literal

if TYPE_CHECKING:
    from ..shader import Shader


_SUPPORT_ANISOTROPIC_FILTER = glInitTextureFilterAnisotropicEXT()
if _SUPPORT_ANISOTROPIC_FILTER:
    _MAX_ANISOTROPY = gl.glGetFloatv(GL_TEXTURE_MAX_ANISOTROPY_EXT)
else:
    _MAX_ANISOTROPY = None

def __SUPPORT_GL_CUDA_SHARE__():
    return GetOrAddGlobalValue('__SUPPORT_GL_CUDA_SHARE__', defaultValue=True)

@attrs(eq=False, repr=False)
class Texture(ResourcesObj):
    '''Base class for texture.'''

    BaseClsName = 'Texture'
    '''The base class name of all texture sub classes'''

    width: int = attrib(default=512)
    '''The width of the texture'''
    height: int = attrib(default=512)
    '''The height of the texture'''
    format: TextureFormat = attrib(default=TextureFormat.RGB)
    '''The specific OpenGL format of the texture.'''
    data: Optional[bytes] = attrib(default=None)
    '''The data of the texture. It should be a bytes object.'''
    min_filter: TextureFilter = attrib(default=TextureFilter.LINEAR_MIPMAP_LINEAR)
    '''The minification filter of the texture.'''
    mag_filter: TextureFilter = attrib(default=TextureFilter.LINEAR)
    '''The magnification filter of the texture.'''
    s_wrap: TextureWrap = attrib(default=TextureWrap.REPEAT)
    '''The wrap mode of the texture in s direction.'''
    t_wrap: TextureWrap = attrib(default=TextureWrap.REPEAT)
    '''The wrap mode of the texture in t direction.'''
    internal_format: TextureInternalFormat = attrib(default=None)
    '''The specific OpenGL internal format of the texture. If not given, it will be set to the default internal format of the format.
    During creation, you can set this to `None`, and it will be set to the default internal format of the format.'''
    data_type: TextureDataType = attrib(default=None)
    '''The specific OpenGL data type of the texture. If not given, it will be set to the default data type of the format.
    During creation, you can set this to `None`, and it will be set to the default data type of the format.'''
    share_to_torch: bool = attrib(default=False)
    '''If true, the texture will be shared to torch by using cuda.'''
    _tensor: Optional[Tensor] = attrib(default=None, alias='tensor')
    '''The tensor of the texture. It could be None if the texture is not shared to torch.'''
    texID: Optional[int] = attrib(default=None)
    '''The opengl texture id. It could be None if the texture is not loaded.'''
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        
        if self.internal_format is None:
            self.internal_format = self.format.default_internal_format
        if self.data_type is None:
            self.data_type = self.internal_format.value.default_data_type
        
        if self.share_to_torch and not __SUPPORT_GL_CUDA_SHARE__():
            if is_dev_mode():
                EngineLogger.warn(f'The data type {self.data_type} does not support tensor. Cannot share to torch. `share_to_torch` will be set to False.')
                self.share_to_torch = False
   
    @property
    def loaded(self)->bool:
        '''If true, the texture has been sent to GPU.'''
        return self.texID is not None
    
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
    def channel_count(self)->int:
        '''The channel count of the texture.'''
        return self.format.channel_count
   
    @Overload
    def bind(self, slot:int, uniformID):    # type: ignore
        '''
        Bind texture to a slot and set shader uniform.
        Make sure you have used shader before calling this function.
        '''
        if self.texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind')
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)   # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texID)
        gl.glUniform1i(uniformID, slot)
        
    def unbind(self, slot:int):
        if self.texID is None:
            return
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)   # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
    @Overload
    def bind(self):
        '''call `glBindTexture` to bind the texture to the current context.'''
        if self.texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind')
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texID)

    @Overload
    def bind(self, slot:int, name:str, shader:'Shader'):
        '''Use shader, and bind texture to a slot and set shader uniform with your given name.'''
        shader.useProgram()
        self.bind(slot, shader.getUniformID(name))

    def _init_tensor(self):
        '''The tensor of the texture. It could be None if the texture is not shared to torch.'''
        assert self.share_to_torch, 'This texture is not set to be sharing to torch.'
        assert self.texID is not None, 'This texture is not yet sent to GPU.'
        
        if self._tensor is None:
            self._tensor = torch.zeros((self.height, self.width, self.channel_count), 
                                        dtype=self.data_type.value.torch_dtype, 
                                        device=f"cuda")
        if not GetOrAddGlobalValue('__SUPPORT_GL_CUDA_SHARE__', True):
            SetGlobalValue('__SUPPORT_GL_CUDA_SHARE__', True)
            return
        try:
            self._cuda_buffer = pycuda.gl.RegisteredImage(int(self.texID),
                                                          int(gl.GL_TEXTURE_2D), 
                                                          pycuda.gl.graphics_map_flags.NONE)
        except Exception as e:
            SetGlobalValue('__SUPPORT_GL_CUDA_SHARE__', False)
            EngineLogger.warning(f'Warning: Texture {self._name} failed to register image. Error: {e}')
            return
        except:
            SetGlobalValue('__SUPPORT_GL_CUDA_SHARE__', False)
            EngineLogger.warning(f'Warning: Texture {self._name} failed to register image. Error: {sys.exc_info()[0]}')
            return
        
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
        assert self.texID, 'This texture is not yet sent to GPU.'
        
        if not update and self._tensor is not None:
            if flip:
                return self._tensor.flip(0)
            return self._tensor
        
        if self.share_to_torch and __SUPPORT_GL_CUDA_SHARE__():
            mapping = self._cuda_buffer.map()
            array = mapping.array(0, 0)
            self._to_tensor_copier.set_src_array(array)
            self._to_tensor_copier(aligned=False)  # this will update self._tensor by copying data from GPU to GPU
            torch.cuda.synchronize()
            mapping.unmap()
        else:
            numpy_data = self.numpy_data(flipY=False)
            if numpy_data.dtype == np.uint32:
                numpy_data = numpy_data.astype(np.int32)    # torch doesn't support uint32
            self._tensor = torch.from_numpy(numpy_data).to(f"cuda:{self.engine.RenderManager.TargetDevice}")
        
        if flip:
            return self._tensor.flip(0) # type: ignore
        return self._tensor # type: ignore
    
    def load(self):
        '''
        Create texture in GPU and send data.
        
        Note:
            This method could still be called even when self.data is None. 
            It means you are creating an empty texture(usually for FBO).
        '''
        if self.loaded:
            return
        
        super().load()
        
        self.texID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texID)
        
        data = self.data
        if not data:
            data = np.zeros((self.height, self.width, self.format.channel_count), 
                            dtype=self.data_type.value.numpy_dtype
                            ).tobytes()
        
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
            
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def clear(self):
        '''
        Clear all data and release GPU memory if it has been loaded.
        This is a default implementation, you may need to override it.
        '''
        super().clear() # just for printing logs
        if self.texID is not None:
            try:
                buffer = np.array([self.texID], dtype=np.uint32)
                gl.glDeleteTextures(1, buffer)
            except OpenGL.error.GLError as glErr:
                if glErr.err == 1282:
                    pass
                else:
                    EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(glErr))
            except Exception as err:
                EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(err))
            self.texID = None
            
        self.data = None
        self.tensor = None  # type: ignore

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
            return
            
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
            
            if self.share_to_torch and data.device.type == 'cuda' and __SUPPORT_GL_CUDA_SHARE__():
                
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
                data = data.cpu().numpy().astype(self.data_type.value.numpy_dtype).tobytes()
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
             internal_format: Optional[TextureInternalFormat]=None,
             data_type: Optional[TextureDataType]=None,
             share_to_torch: bool = False,
             **kwargs,
             )->'Texture':
        '''Create & load a texture from a file path. The format will be determined by the file extension.'''
        # default load by PIL
        if is_dev_mode():
            EngineLogger.debug('Loading texture by path:' + str(path) + '...')
        
        image = Image.open(path).convert(format.PIL_convert_mode)
        data = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        height = image.height
        width = image.width
        image.close()
        
        internal_format = internal_format or format.default_internal_format
        data_type = data_type or internal_format.value.default_data_type
        
        texture = cls(name=name, 
                      format=format, 
                      width=width,
                      height=height,
                      min_filter=min_filter, 
                      mag_filter =mag_filter, 
                      data=data,
                      s_wrap=s_wrap, 
                      t_wrap=t_wrap,
                      internal_format=internal_format,
                      data_type=data_type,
                      share_to_torch=share_to_torch,
                      **kwargs)
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
                         internal_format: Optional[TextureInternalFormat] = None,
                         data_type: Optional[TextureDataType] = None,
                         share_to_torch: bool = False,
                         )->'Texture':
        '''Create an empty texture with a fill color as image.'''
        if isinstance(fill, (int, float)):
            tex_format = TextureFormat.RED
            if not internal_format and isinstance(fill, int):
                internal_format = TextureInternalFormat.RED_16UI
                
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
        
        internal_format = internal_format or tex_format.default_internal_format
        data_type = data_type or internal_format.value.default_data_type
        
        texture = Texture(name=name,
                          width=width,
                          height=height,
                          format=tex_format,
                          data = fake_image.tobytes(),
                          min_filter=min_filter,
                          mag_filter=mag_filter,
                          s_wrap=s_wrap,
                          t_wrap=t_wrap,
                          internal_format=internal_format,
                          data_type=data_type,
                          share_to_torch=share_to_torch)
        
        return texture
    
    @staticmethod
    def CreateNoiseTex(name:Optional[str]=None,
                       height: int = 512,
                       width: int = 512,
                       channel_count = 4,   # same as latent size
                       data_size: Literal[16, 32] = 32,
                       min_filter:TextureFilter=TextureFilter.NEAREST,
                       mag_filter:TextureFilter=TextureFilter.NEAREST,
                       s_wrap:TextureWrap=TextureWrap.REPEAT,
                       t_wrap:TextureWrap=TextureWrap.REPEAT,
                       internal_format: Optional[TextureInternalFormat] = None,
                       share_to_torch: bool = False,
                       )->'Texture':
        '''
        Create a texture which is filled with random gaussian noise in each pixel.
        
        Args:
            - name: The name of the texture.
            - height: The height of the texture. Default is 64(same as latent size when origin width is 512)
            - width: The width of the texture. Default is 64(same as latent size when origin height is 512)
            - sigma: The standard deviation of the gaussian noise.
            - channel_count: The channel count of the texture. It should be in [1, 4].
                             Default is 4, which is same as latent size.
            - data_size: The data size of the texture. It should be float16 or float32.
            - min_filter: The minification filter of the texture. Default is NEAREST(since it's noise texture, should not be interpolated)
            - mag_filter: The magnification filter of the texture. Default is NEAREST(since it's noise texture, should not be interpolated)
            - s_wrap: The wrap mode of the texture in s direction.
            - t_wrap: The wrap mode of the texture in t direction.
            - internal_format: The specific OpenGL internal format of the texture. If not given, it will be set to the default internal format of the format.
            - share_to_torch: If true, the texture will be shared to torch by using cuda.
        '''
        
        if channel_count < 1 or channel_count > 4:
            raise Exception(f'Invalid channel count: {channel_count}. It should be in [1, 4].')
        
        assert data_size in (16, 32), 'Invalid data size: {}'.format(data_size)
        dtype = {16: torch.float16, 32: torch.float32}[data_size]
        data = torch.randn((height, width, channel_count), dtype=dtype).numpy()
        
        if not internal_format:
            if channel_count == 1:
                internal_format = TextureInternalFormat.RED_32F if data_size == 32 else TextureInternalFormat.RED_16F
            elif channel_count == 2:
                internal_format = TextureInternalFormat.RG32F if data_size == 32 else TextureInternalFormat.RG16F
            elif channel_count == 3:
                internal_format = TextureInternalFormat.RGB32F if data_size == 32 else TextureInternalFormat.RGB16F
            elif channel_count == 4:
                internal_format = TextureInternalFormat.RGBA32F if data_size == 32 else TextureInternalFormat.RGBA16F
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
                      internal_format=internal_format,
                      data_type=TextureDataType.FLOAT if data_size == 32 else TextureDataType.HALF,
                      share_to_torch=share_to_torch)
        return tex

    def __deepcopy__(self):
        '''
        For `custom_deep_copy` method in common_utils.type_utils.
        Cloning a texture will only clone the tensor(if it has). The texture will still point to the same OpenGL texture.
        '''
        tex = self.__class__(name=None,
                             width=self.width,
                             height=self.height,
                             format=self.format,
                             data=self.data,
                             min_filter=self.min_filter,
                             mag_filter=self.mag_filter,
                             s_wrap=self.s_wrap,
                             t_wrap=self.t_wrap,
                             internal_format=self.internal_format,
                             data_type=self.data_type,
                             share_to_torch=self.share_to_torch,
                             tensor=self._tensor.clone() if self._tensor is not None else None,
                             texID = self.texID)
        return tex
        
        
__all__ = ['Texture']