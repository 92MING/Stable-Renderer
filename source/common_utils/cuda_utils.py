import cuda.cudart as cudart
import cuda.cuda as cuda
import OpenGL.GL as gl

from OpenGL.GL import Constant as GLConst
from enum import IntEnum
from functools import wraps
from typing import (Callable, TypeVar, Tuple, TypeAlias, Literal, ParamSpec, Union, Sequence, 
                    TYPE_CHECKING, Optional)
if TYPE_CHECKING:
    from engine.static.texture import Texture


P = ParamSpec('P')
T = TypeVar('T')

CudaErrType: TypeAlias = cudart.cudaError_t

class CudaError(Exception): ...


__all__ = ['CudaError',]

def _format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}(code={int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )

def _cuda_func_wrapper(func: Callable[P, Tuple[CudaErrType, T]])->Callable[P, T]:
    '''help raising error when cuda function returns an error code'''
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs)->T:
        err, result = func(*args, **kwargs)
        if err:
            raise CudaError(f'Cuda error occurred in `{func.__name__}` with error: {_format_cudart_err(err)}')
        return result
    return wrapper

# region device
@_cuda_func_wrapper
def get_cuda_device_count()->Tuple[CudaErrType, int]:
    '''return the number of devices supporting CUDA'''
    return cudart.cudaGetDeviceCount()

@_cuda_func_wrapper
def get_cuda_gl_devices()->Tuple[CudaErrType, Tuple[int]]:
    '''return the list of devices supporting OpenGL interoperability'''
    device_count = get_cuda_device_count()
    err, _, lst = cudart.cudaGLGetDevices(device_count, cudart.cudaGLDeviceList.cudaGLDeviceListAll)
    return err, tuple(set(lst)) # type: ignore

@_cuda_func_wrapper
def set_cuda_device(device: int)->Tuple[CudaErrType, None]:
    '''set the current device to device'''
    err, = cudart.cudaSetDevice(device)
    return (err, None)
    
@_cuda_func_wrapper
def get_cuda_device()->Tuple[CudaErrType, int]:
    '''Returns which device is currently being used'''
    err, device = cudart.cudaGetDevice()
    return (err, device)

@_cuda_func_wrapper
def cuda_init_device(device: int,
                     deviceFlags: int = 0,
                     flags: int = 0,
                     )->Tuple[CudaErrType, None]:
    '''initialize the device'''
    err, = cudart.cudaInitDevice(device, deviceFlags, flags)
    return (err, None)

__all__.extend(['get_cuda_device', 'set_cuda_device', 'cuda_init_device', 'get_cuda_device_count', 'get_cuda_gl_devices'])
# endregion

# region OpenGL related
class GLRegisterFlag(IntEnum):
    NONE = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone
    '''Specifies no hints about how this resource will be used. 
      It is therefore assumed that this resource will be read from and written to by CUDA. 
      This is the default value.'''
    READ_ONLY = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
    '''Specifies that CUDA will not write to this resource.'''
    WRITE_DISCARD = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    '''CUDA will not read from this resource and will write over the entire contents of the resource, 
      so none of the data previously stored in the resource will be preserved.'''
    LOAD_STORE = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore
    '''CUDA will bind this resource to a surface reference.''' 
    TEXTURE_GATHER = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsTextureGather
    '''CUDA will perform texture gather operations on this resource.'''


_register_gl_image_map = {
    'texture2d': gl.GL_TEXTURE_2D,
    'texturecube': gl.GL_TEXTURE_CUBE_MAP,
    'texture2darray': gl.GL_TEXTURE_2D_ARRAY,
}

@_cuda_func_wrapper
def cuda_register_gl_image(image: Union[int, 'Texture'],
                           target: Union[Literal['texture2D', 'textureCube', 'texture2DArray'], GLConst] = 'texture2D',
                           flags:  Union[int, GLRegisterFlag, Sequence[GLRegisterFlag]] = GLRegisterFlag.NONE,
                           )->Tuple[CudaErrType, cudart.cudaGraphicsResource_t]:
    '''return the pointer to the resource'''
    if isinstance(target, str):
        target = target.lower() # type: ignore
        if target not in _register_gl_image_map:
            raise ValueError(f'Invalid target: {target}, must be one of {list(_register_gl_image_map)}')
        target = _register_gl_image_map[target]
    elif not isinstance(target, GLConst):
        raise TypeError(f'target must be GLConst or one of {list(_register_gl_image_map)}')
    
    if isinstance(flags, Sequence):
        flags = sum(flags)
    if hasattr(image, '_BaseName') and image._BaseName == 'Texture':  # type: ignore
        image = image.textureID  # type: ignore
        
    return cudart.cudaGraphicsGLRegisterImage(image, target, flags)
    
@_cuda_func_wrapper
def cuda_map_resources(resource: Union[cudart.cudaGraphicsResource_t, 'Texture', int],
                       count: int =1,
                       stream:Optional[cudart.cudaStream_t] = None)->Tuple[CudaErrType, None]:
    '''
    Maps the count graphics resources in resources for access by CUDA.
    
    Args:
        resources: A list of count resources to map for access by CUDA.
        count: The number of resources to map for access by CUDA.
        stream: Stream for synchronization
    '''
    if hasattr(resource, '_BaseName') and resource._BaseName == 'Texture':  # type: ignore
        resource = resource._cuda_resource_ptr  # type: ignore
        if resource is None:
            raise ValueError('The texture is not registered as a CUDA resource')
        
    err, = cudart.cudaGraphicsMapResources(count, resource, stream)
    return err, None

@_cuda_func_wrapper
def cuda_get_mapped_pointer(resource: Union[cudart.cudaGraphicsResource_t, 'Texture', int])->Tuple[CudaErrType, Tuple[int, int]]:
    '''
    Returns the pointer through which a CUDA graphics resource is mapped.
    
    Args:
        resource: The texture to map for access by CUDA.
        
    Return:
        devicePtr: The base pointer of the mapped graphics resource.
        size: Returned size of the buffer accessible starting at *devPtr
    '''
    if hasattr(resource, '_BaseName') and resource._BaseName == 'Texture':  # type: ignore
        resource = resource._cuda_resource_ptr  # type: ignore
        if resource is None:
            raise ValueError('The texture is not registered as a CUDA resource')
    
    err, devicePtr, size = cudart.cudaGraphicsResourceGetMappedPointer(resource)
    return (err, (devicePtr, size))

@_cuda_func_wrapper
def cuda_get_mapped_array(resource: Union[cudart.cudaGraphicsResource_t, 'Texture', int],
                          arrayIndex:int = 0,
                          mipLevel:int = 0,
                          )->Tuple[CudaErrType, cudart.cudaArray_t]:
    '''
    Returns the array through which a CUDA graphics resource is mapped.
    
    Args:
        resource: The texture to map for access by CUDA.
        arrayIndex: The index of the array to map for access by CUDA.
        mipLevel: The mipmap level to map for access by CUDA.
    '''
    if hasattr(resource, '_BaseName') and resource._BaseName == 'Texture':  # type: ignore
        resource = resource._cuda_resource_ptr  # type: ignore
        if resource is None:
            raise ValueError('The texture is not registered as a CUDA resource')
    
    return cudart.cudaGraphicsSubResourceGetMappedArray(resource, arrayIndex, mipLevel)

@_cuda_func_wrapper
def cuda_get_array_info(array: Union[cudart.cudaArray_t, int])->Tuple[CudaErrType, Tuple[cudart.cudaChannelFormatDesc, cudart.cudaExtent, int]]:
    '''
    Returns the channel descriptor and size of the CUDA array.
    
    Args:
        array: The CUDA array to get the channel descriptor and size of.
    '''
    err, channel_format, cuda_extent, flags = cudart.cudaArrayGetInfo(array)
    return err, (channel_format, cuda_extent, flags)


__all__.extend(['GLRegisterFlag', 'cuda_register_gl_image', 
                'cuda_map_resources', 'cuda_get_mapped_array', 'cuda_get_mapped_pointer', 'cuda_get_array_info'])
# endregion