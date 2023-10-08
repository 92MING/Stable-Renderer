import numpy as np
from static.data_types.vector import Vector
from typing import List, Union

class Vertex:
    def __init__(self, pos:Vector, normal:Vector=None, uv:Vector=None):
        if not (pos.dtype == normal.dtype == uv.dtype):
            raise TypeError('dtype of pos, normal and uv must be the same')
        self.pos:Vector = pos
        self.normal:Vector = normal
        self.uv:Vector = uv

    @property
    def has_normal(self):
        return self.normal is not None
    @property
    def has_uv(self):
        return self.uv is not None
    @property
    def size(self):
        '''memory size'''
        return self.pos.size + (self.normal.size if self.has_normal else 0) + (self.uv.size if self.has_uv else 0)

class VertexList(list):

    @property
    def size(self):
        '''memory size'''
        if len(self) == 0:
            return 0
        return self[0].size * len(self)
    @property
    def interval_size(self):
        '''size of a single vertex'''
        if len(self) == 0:
            return 0
        return self[0].size
    @property
    def has_normal(self):
        if len(self) == 0:
            return False
        return self[0].has_normal
    @property
    def has_uv(self):
        if len(self) == 0:
            return False
        return self[0].has_uv

    def append(self, v:Vertex):
        if not isinstance(v, Vertex):
            raise TypeError(f'expected Vertex, got {type(v)}')
        super().append(v)

    def append_vertex(self, pos:Vector, normal:Vector=None, uv:Vector=None):
        self.append(Vertex(pos, normal, uv))

    def toArray(self)->np.ndarray:
        '''Prepare vertex data for OpenGL EBO'''
        if len(self) == 0:
            raise ValueError('VertexList is empty')
        hasNormal = self.has_normal
        hasUV = self.has_uv
        data = []
        for v in self:
            data.extend(v.pos.tolist())
            if hasNormal:
                data.extend(v.normal.tolist())
            if hasUV:
                data.extend(v.uv.tolist())
        return np.array(data, dtype=self[0].pos.dtype)

__all__ = ['Vertex', 'VertexList']