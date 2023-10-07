import numpy as np
from functools import partial
from typing import Sequence
from static.data_types.vector import Vector
import warnings

def _getMatrix(type, *args):
    if type == int:
        warnings.warn('Note that int matrix is not supported for opengl uniform.')
        type = np.int32
    return np.matrix(*args, dtype=type).view(Matrix)

class Matrix(np.matrix):
    def __new__(cls, *args, **kwargs):
        '''When type is not specified, it will be inferred from the first element of args.'''
        dtype = kwargs.get('dtype', None)
        if dtype is None:
            if not isinstance(args[0][0], Sequence):
                args = [args,]
            dtype = type(args[0][0][0])
            if dtype ==float:
                dtype = np.float32
            elif dtype == int:
                warnings.warn('Note that int matrix is not supported for opengl uniform.')
                dtype = np.int32
        return np.matrix(*args, dtype=dtype).view(cls)
    def __class_getitem__(cls, item):
        if item == float:
            item = np.float32
        return partial(_getMatrix, item)
    @property
    def size(self):
        '''Return memory size in bytes'''
        return self.itemsize * super().size

    @staticmethod
    def Identity(n:int, type=np.float32):
        return Matrix[type](np.identity(n))

    # region transformation
    @staticmethod
    def RotationX(angle, radian=False):
        '''4x4 matrix'''
        t = type(angle)
        if t in (int, float):
            t = np.float32
        if not radian:
            angle = np.radians(angle)
        return Matrix[t]([[1, 0, 0, 0],
                         [0, np.cos(angle), -np.sin(angle), 0],
                         [0, np.sin(angle), np.cos(angle), 0],
                         [0, 0, 0, 1]])
    @staticmethod
    def RotationY(angle, radian=False):
        '''4x4 matrix'''
        t = type(angle)
        if t in (int, float):
            t = np.float32
        if not radian:
            angle = np.radians(angle)
        return Matrix[t]([[np.cos(angle), 0, np.sin(angle), 0],
                         [0, 1, 0, 0],
                         [-np.sin(angle), 0, np.cos(angle), 0],
                         [0, 0, 0, 1]])
    @staticmethod
    def RotationZ(angle, radian=False):
        '''4x4 matrix'''
        t = type(angle)
        if t in (int, float):
            t = np.float32
        if not radian:
            angle = np.radians(angle)
        return Matrix[t]([[np.cos(angle), -np.sin(angle), 0, 0],
                           [np.sin(angle), np.cos(angle), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    @staticmethod
    def Rotation(x, y, z, radian=False):
        '''rotation order is z, x, y. Note that this is 4x4 matrix'''
        if not radian:
            z, x, y = np.radians(z), np.radians(x), np.radians(y)
        return Matrix.RotationZ(z, radian=radian) * Matrix.RotationX(x, radian=radian) * Matrix.RotationY(y, radian=radian)

    @staticmethod
    def Translation(x, y, z):
        t = type(x)
        if t in (int, float):
            t = np.float32
        return Matrix[t]([[1, 0, 0, x],
                          [0, 1, 0, y],
                          [0, 0, 1, z],
                          [0, 0, 0, 1]])
    @staticmethod
    def Scale(x, y, z):
        t = type(x)
        if t in (int, float):
            t = np.float32
        return Matrix[t]([[x, 0, 0, 0],
                          [0, y, 0, 0],
                          [0, 0, z, 0],
                          [0, 0, 0, 1]])

    @staticmethod
    def Transformation(translation, rotation, scale):
        '''TRS'''
        t = Matrix.Translation(translation[0], translation[1], translation[2])
        r = Matrix.Rotation(rotation[0], rotation[1], rotation[2])
        s = Matrix.Scale(scale[0], scale[1], scale[2])
        return t * r * s
    # endregion


    # region projection
    @staticmethod
    def Orthographic(left, right, bottom, top, near, far, type=np.float32):
        return Matrix[type]([[2 / (right - left), 0, 0, -(right + left) / (right - left)],
                             [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                             [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                             [0, 0, 0, 1]])
    @staticmethod
    def Perspective(fov, aspect, near, far, type=np.float32):
        f = 1 / np.tan(np.radians(fov / 2))
        return Matrix[type]([[f / aspect, 0, 0, 0],
                             [0, f, 0, 0],
                             [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                             [0, 0, -1, 0]])
    # endregion

    # region lookat
    @staticmethod
    def LookAt(eye, center, up, type=np.float32):
        '''
        :param eye: eye position
        :param center: center position
        :param up: up vector
        :return:
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
        return Matrix[type]([[s[0], s[1], s[2], -s.dot(eye)],
                            [u[0], u[1], u[2], -u.dot(eye)],
                            [-f[0], -f[1], -f[2], f.dot(eye)],
                            [0, 0, 0, 1]])
    # endregion

__all__ = ['Matrix']