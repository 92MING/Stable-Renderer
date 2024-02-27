import numba
import numpy as np
from typing import Callable

def pre_compile_njit(*args):
    '''Wrap a function with `njit` and run it immediately with the given arguments for pre-compilation'''
    def wrapper(func: Callable):
        njit_func = numba.njit()(func)
        njit_func(*args)
        return njit_func
    return wrapper

_triple_zero = np.array([0.0, 0.0, 0.0])
_double_zero = np.array([0.0, 0.0])
@pre_compile_njit(_triple_zero, _triple_zero, _triple_zero, _double_zero, _double_zero, _double_zero)
def calculate_tangent(center_point_xyz: np.ndarray,
                      left_point_xyz: np.ndarray,
                      right_point_xyz: np.ndarray,
                      center_point_uv: np.ndarray,
                      left_point_uv: np.ndarray,
                      right_point_uv) -> np.ndarray:
    '''
    calculate tangent.
    :param center_point_xyz: center point of the triangle in 3D space
    :param left_point_xyz: left point of the triangle in 3D space
    :param right_point_xyz: right point of the triangle in 3D space
    :param center_point_uv: center point of the triangle in UV space
    :param left_point_uv: left point of the triangle in UV space
    :param right_point_uv: right point of the triangle in UV space
    :return: tangent
    '''
    edge1 = left_point_xyz - center_point_xyz
    edge2 = right_point_xyz - center_point_xyz
    deltaUV1 = left_point_uv - center_point_uv
    deltaUV2 = right_point_uv - center_point_uv

    d = deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1]
    if d != 0:
        f = 1.0 / d
    else:
        f = 1.0
    return np.array([f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
                     f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
                     f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])])



__all__ = ['calculate_tangent']