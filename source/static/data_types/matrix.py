import numpy as np
from static.data_types.vector import Vector
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
        return cls.RotationZ(z, radian=radian) @ cls.RotationX(x, radian=radian) @ cls.RotationY(y, radian=radian)

    @classmethod
    def Rotation_around_axis(cls, axis, angle, radian=False):
        '''rotation around axis'''
        if not radian:
            angle = np.radians(angle)
        axis = Vector[cls.type()](axis)
        axis = axis.normalize
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
            eye = Vector(eye)
        if isinstance(center, (list, tuple)):
            center = Vector(center)
        if isinstance(up, (list, tuple)):
            up = Vector(up)
        zaxis = (eye - center).view(Vector).normalize
        xaxis = up.cross(zaxis).view(Vector).normalize
        yaxis = zaxis.cross(xaxis).view(Vector).normalize
        return cls([[xaxis.x, xaxis.y, xaxis.z, -xaxis.dot(eye)],
                    [yaxis.x, yaxis.y, yaxis.z, -yaxis.dot(eye)],
                    [zaxis.x, zaxis.y, zaxis.z, -zaxis.dot(eye)],
                    [0, 0, 0, 1]], dtype=cls.type())
    # endregion


__all__ = ['Matrix']

if __name__ == '__main__':
    modelMatrix = Matrix.Transformation([0.0, 0.0, -5], [0, 90, 0], [1.0, 1.0, 1.0])
    m = (Vector([0,0,0,1]) * modelMatrix)
    print(m, m.shape)