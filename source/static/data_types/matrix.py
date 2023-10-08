import numpy as np
from functools import partial
from typing import Sequence
from static.data_types.vector import Vector
import warnings
from utils.base_clses import SingleGenericClass


def _getMatrix(type, *args):
    return np.matrix(*args, dtype=type).view(Matrix)


class Matrix(SingleGenericClass, np.matrix):
    '''Default type is np.float32'''
    DEFAULT_TYPE = np.float32

    def __new__(cls, *args, **kwargs):
        if len(args) >= 2:
            dtype = args[1]
            if dtype == int:
                dtype = np.int32
            elif dtype == float:
                dtype = np.float32
            args = list(args)
            args[1] = dtype
            return np.array(*args, **kwargs).view(cls)
        else:
            dtype = kwargs.pop('dtype', cls.type())
            return np.array(*args, dtype=dtype, **kwargs).view(cls)

    @classmethod
    def type(cls):
        if cls._type is None:
            if cls.DEFAULT_TYPE is not None:
                return cls.DEFAULT_TYPE
            raise TypeError(
                f'{cls.__name__} has no generic type. Please use {cls.__name__}[type] to get a generic class')
        if cls._type == float:
            return np.float32
        elif cls._type == int:
            return np.int32
        return cls._type

    @property
    def size(self):
        '''Return memory size in bytes'''
        return self.itemsize * super().size

    @classmethod
    def Identity(cls, n: int):
        return cls(np.identity(n, dtype=cls.type()))

    # region transformation
    @classmethod
    def RotationX(cls, angle, radian=False):
        '''4x4 matrix'''
        if not radian:
            angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        return cls([[1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def RotationY(cls, angle, radian=False):
        '''4x4 matrix'''
        if not radian:
            angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        return cls([[c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def RotationZ(cls, angle, radian=False):
        '''4x4 matrix'''
        if not radian:
            angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        return cls([[c, -s, 0, 0],
                   [s, c, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def Rotation(cls, x, y, z, radian=False):
        '''rotation order is z, x, y. Note that this is 4x4 matrix'''
        if not radian:
            z, x, y = np.radians(z), np.radians(x), np.radians(y)
        return cls.RotationZ(z, radian=radian) @ cls.RotationX(x, radian=radian) @ cls.RotationY(y, radian=radian)

    @classmethod
    def Rotation_around_axis(cls, axis, angle, radian=False):
        '''rotation around axis'''
        if not radian:
            angle = np.radians(angle)
        axis = Vector(axis, dtype=cls.type())
        axis = axis.normalized()
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        return cls([[t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
                    [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
                    [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def Translation(cls, x, y, z):
        return cls([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def Scale(cls, x, y, z):
        return cls([[x, 0, 0, 0],
                    [0, y, 0, 0],
                    [0, 0, z, 0],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def Transformation(cls, translation, rotation, scale):
        '''TRS'''
        t = cls.Translation(translation[0], translation[1], translation[2])
        r = cls.Rotation(rotation[0], rotation[1], rotation[2])
        s = cls.Scale(scale[0], scale[1], scale[2])
        return t * r * s
    # endregion

    # region projection
    @classmethod
    def Orthographic(cls, left, right, bottom, top, near, far):
        return cls([[2 / (right - left), 0, 0, -(right + left) / (right - left)],
                    [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                    [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                    [0, 0, 0, 1]], dtype=cls.type())

    @classmethod
    def Perspective(cls, fov, aspect, near, far, radian=False):
        if not radian:
            fov = np.radians(fov)
        f = 1 / np.tan(fov / 2)
        return cls([[f / aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                    [0, 0, -1, 0]], dtype=cls.type())
    # endregion

    # region lookat
    @classmethod
    def LookAt(cls, eye, center, up):
        '''
        :param eye: eye position
        :param center: center position
        :param up: up vector
        :return: 4x4 View matrix
        '''
        if isinstance(eye, (list, tuple)):
            eye = np.array(eye)
        if isinstance(center, (list, tuple)):
            center = np.array(center)
        if isinstance(up, (list, tuple)):
            up = np.array(up)
        f = (center - eye).view(Vector).normalize
        s = f.cross(up).view(Vector).normalize
        u = s.cross(f)
        return cls([[s[0], s[1], s[2], -s @ eye],
                    [u[0], u[1], u[2], -u @ eye],
                    [-f[0], -f[1], -f[2], f @ eye],
                    [0, 0, 0, 1]], dtype=cls.type())
    # endregion


__all__ = ['Matrix']
