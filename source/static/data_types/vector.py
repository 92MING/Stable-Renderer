import numpy as np
from utils.base_clses import SingleGenericClass
from typing import Sequence
from .base_types import *

class Vector(np.array, SingleGenericClass):
    def __new__(cls, *args):
        if cls._type is None:
            if len(values) == 0:
                raise ValueError('Vector must have at least one value')
            if hasattr(values[0], '__qualname__') and values[0].__qualname__ == 'Vector':
                t = values[0]._type
            elif isinstance(values[0], Sequence) or isinstance(values[0], np.ndarray):
                t = type(values[0][0])
            else:
                t = type(values[0])
            if isinstance(t, int):
                t = Int
            elif isinstance(t, float):
                t = Float
            return cls[t](*values)
        else:
            return super().__new__(cls)
    def __init__(self, *values):
        if len(values) == 1:
            if isinstance(values[0], Vector):
                self._values = np.array(values[0]._values, dtype=self._type)
            elif isinstance(values[0], np.ndarray) or isinstance(values[0], Sequence):
                self._values = np.array(values[0], dtype=self._type)
            else:
                self._values = np.array([values[0]], dtype=self._type)
        else:
            self._values = np.array(values, dtype=self._type)
    def __repr__(self):
        return f'Vector{self.dimension}<{self._type}>({", ".join(map(str, self._values))})'
    def __str__(self):
        return str(self._values.tolist())
    def __getitem__(self, item):
        return self._values[item]
    def __class_getitem__(cls, item):
        if item == int:
            item = Int
        elif item == float:
            item = Float
        return super().__class_getitem__(item)

    @classmethod
    def Zero(cls, dimension:int):
        if cls.type() is None:
            raise ValueError('Cannot create a zero vector with no type')
        return cls(np.zeros(dimension, dtype=cls._type))

    def dot(self, other:'Vector'):
        if self.dimension != other.dimension:
            raise ValueError(f'Vector{self.dimension} and Vector{other.dimension} cannot dot')
        return np.dot(self._values, other._values)
    def cross(self, other:'Vector'):
        if self.dimension != other.dimension:
            raise ValueError(f'Vector{self.dimension} and Vector{other.dimension} cannot cross')
        vec = np.cross(self._values, other._values)
        return Vector(vec)
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        else:
            return Vector(self._values * other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        return Vector(self._values / other)
    def __add__(self, other):
        return Vector(self._values + other._values)
    def __sub__(self, other):
        return Vector(self._values - other._values)
    def __neg__(self):
        return Vector(-self._values)
    def __abs__(self):
        return self.norm
    def __eq__(self, other):
        return np.all(self._values == other._values)
    def __iter__(self):
        return iter(self._values)
    def __len__(self):
        return len(self._values)

    @property
    def size(self):
        '''memory size in bytes'''
        return self._values.nbytes
    @property
    def norm(self)->float:
        return np.linalg.norm(self._values)
    @property
    def normalize(self)->'Vector':
        return Vector(self._values / self.norm)
    @property
    def dimension(self)->int:
        return len(self._values)

    @property
    def x(self)->float:
        return self._values[0]
    @property
    def y(self)->float:
        if len(self._values) < 2:
            raise IndexError('Vector has no y')
        return self._values[1]
    @property
    def z(self)->float:
        if len(self._values) < 3:
            raise IndexError('Vector has no z')
        return self._values[2]
    @property
    def w(self)->float:
        if len(self._values) < 4:
            raise IndexError('Vector has no w')
        return self._values[3]
    @property
    def u(self)->float:
        return self.x
    @property
    def v(self)->float:
        if len(self._values) < 2:
            raise IndexError('Vector has no v')
        return self.y
    @property
    def xy(self)->'Vector':
        return Vector(self._values[:2])
    @property
    def yz(self)->'Vector':
        return Vector(self._values[1:3])
    @property
    def uv(self)->'Vector':
        return Vector(self._values[:2])

    @property
    def _xyz(self) -> 'Vector':
        return self._values[:3]
