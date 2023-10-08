import numpy as np
from typing import Sequence
from utils.base_clses import SingleGenericClass

class Vector(SingleGenericClass, np.ndarray):
    '''Default type is np.float32'''
    DEFAULT_TYPE = np.float32

    def __new__(cls, *values):
        '''When type is not specified, it will be inferred from the first element of args.'''
        dtype = cls.type()
        if len(values)==1:
            if not isinstance(values[0], (Sequence, np.ndarray)):
                values = (values,)
            else:
                values = values[0]
        return np.array(values, dtype=dtype).view(cls)

    # region magic methods
    def __eq__(self, other):
        result = super().__eq__(other)
        if isinstance(result, np.ndarray):
            return result.all()
        return result
    def __repr__(self):
        '''return as Vector<type>(x, y, z, ...)'''
        return f'Vector<{self.type()}>{tuple(self)}'
    def __str__(self):
        '''return as [x, y, z, ...]'''
        return str(self.tolist())
    # endregion

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
        '''memory size in bytes'''
        return self.nbytes
    @property
    def length(self)->float:
        return np.linalg.norm(self)
    @property
    def normalize(self)->'Vector':
        return Vector[self.type()](self / self.length)
    @property
    def dimension(self)->int:
        return len(self)

    def cross(self, other:'Vector')->'Vector':
        if self.dimension != 3 or other.dimension != 3:
            raise ValueError('Cross product is only defined for 3D vectors')
        return Vector[self.type()](np.cross(self, other))

    @property
    def x(self):
        return self[0]
    @x.setter
    def x(self, val):
        self[0] = val
    @property
    def y(self):
        if len(self) < 2:
            raise IndexError('Vector has no y')
        return self[1]
    @y.setter
    def y(self, val):
        if len(self) < 2:
            raise IndexError('Vector has no y')
        self[1] = val
    @property
    def z(self):
        if len(self) < 3:
            raise IndexError('Vector has no z')
        return self[2]
    @z.setter
    def z(self, val):
        if len(self) < 3:
            raise IndexError('Vector has no z')
        self[2] = val
    @property
    def w(self):
        if len(self) < 4:
            raise IndexError('Vector has no w')
        return self[3]
    @w.setter
    def w(self, val):
        if len(self) < 4:
            raise IndexError('Vector has no w')
        self[3] = val
    @property
    def xyz(self) -> 'Vector':
        return self[:3]
    @property
    def xy(self)->'Vector':
        return Vector[self.type()](self[:2])
    @property
    def yz(self)->'Vector':
        return Vector[self.type()](self[1:3])

    @property
    def u(self):
        return self.x
    @u.setter
    def u(self, val):
        self[0] = val
    @property
    def v(self) -> float:
        if len(self) < 2:
            raise IndexError('Vector has no v')
        return self.y
    @v.setter
    def v(self, val):
        if len(self) < 2:
            raise IndexError('Vector has no v')
        self[1] = val
    @property
    def uv(self) -> 'Vector':
        return Vector[self.type()](self[:2])


__all__ = ['Vector']