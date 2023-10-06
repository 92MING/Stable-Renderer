import numpy as np
from typing import Sequence
from functools import partial

def _getVector(type, *args):
    if isinstance(args[0], Sequence):
        args = (args[0],)
    return np.array(*args, dtype=type).view(Vector)

class Vector(np.ndarray):
    def __new__(cls, *args, **kwargs):
        '''When type is not specified, it will be inferred from the first element of args.'''
        dtype = kwargs.get('dtype', None)
        if dtype is None:
            if not isinstance(args[0], Sequence):
                args = (args,)
            if type(args[0][0])==int:
                dtype = np.int32
            elif type(args[0][0])==float:
                dtype = np.float32
        return np.array(*args, dtype=dtype).view(cls)
    def __class_getitem__(cls, item):
        if item == int:
            item = np.int32
        elif item == float:
            item = np.float32
        return partial(_getVector, item)

    @property
    def size(self):
        '''memory size in bytes'''
        return self.nbytes
    @property
    def length(self)->float:
        return np.linalg.norm(self)
    @property
    def normalize(self)->'Vector':
        return Vector(self / self.length)
    @property
    def dimension(self)->int:
        return len(self)

    @property
    def x(self)->float:
        return self[0]
    @property
    def y(self)->float:
        if len(self) < 2:
            raise IndexError('Vector has no y')
        return self[1]
    @property
    def z(self)->float:
        if len(self) < 3:
            raise IndexError('Vector has no z')
        return self[2]
    @property
    def w(self)->float:
        if len(self) < 4:
            raise IndexError('Vector has no w')
        return self[3]
    @property
    def xyz(self) -> 'Vector':
        return self[:3]
    @property
    def xy(self)->'Vector':
        return Vector(self[:2])
    @property
    def yz(self)->'Vector':
        return Vector(self[1:3])

    @property
    def u(self) -> float:
        return self.x
    @property
    def v(self) -> float:
        if len(self) < 2:
            raise IndexError('Vector has no v')
        return self.y
    @property
    def uv(self) -> 'Vector':
        return Vector(self[:2])

__all__ = ['Vector']