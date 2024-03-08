import cuda.cudart as cudart
import cuda.cuda as cuda

from functools import wraps
from typing import Callable, TypeVar, Tuple, TypeAlias, Concatenate, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

CudaErrType: TypeAlias = cudart.cudaError_t

class CudaError(Exception): ...

__all__ = ['CudaError',]

def _cuda_func_wrapper(func: Callable[P, Tuple[CudaErrType, T]])->Callable[P, T]:
    '''help raising error when cuda function returns an error code'''
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs)->T:
        err, result = func(*args, **kwargs)
        if err:
            raise CudaError(f'Cuda error occurred in {func.__name__} with error: {err.name} ({err})')
        return result
    return wrapper

@_cuda_func_wrapper
def get_cuda_device_count()->Tuple[CudaErrType, int]:
    return cudart.cudaGetDeviceCount()

@_cuda_func_wrapper
def get_cuda_gl_devices()->Tuple[CudaErrType, Tuple[int, Tuple[int]]]:
    device_count = get_cuda_device_count()
    err, count, lst = cudart.cudaGLGetDevices(device_count, cudart.cudaGLDeviceList.cudaGLDeviceListAll)
    return err, tuple(set(lst)) # type: ignore

@_cuda_func_wrapper
def register_gl_buffer(buffer: int)->CudaErrType:
    return cudart.cudaGraphicsGLRegisterBuffer(buffer)
