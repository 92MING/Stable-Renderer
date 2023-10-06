import numpy as np
from .vector import Vector

class Vertex:
    def __init__(self, pos:Vector[3], normal:Vector[3]=None, uv:Vector[2]=None):
        self.pos = pos
        self.normal = normal
        self.uv = uv

class VertexList(list):

    def append(self, v:Vertex):
        if not isinstance(v, Vertex):
            raise TypeError(f'expected Vertex, got {type(v)}')
        super().append(v)

    def appendVertex(self, pos:Vector[3], normal:Vector[3]=None, uv:Vector[2]=None):
        self.append(Vertex(pos, normal, uv))

    def toNpArray(self):
        if len(self) == 0:
            raise ValueError('VertexList is empty')
        return np.array([v.pos._values for v in self], dtype=np.float32)