import os
from static.resourcesObj import ResourcesObj
from utils.global_utils import GetOrAddGlobalValue, SetGlobalValue
import OpenGL.GL as gl
import numpy as np
import ctypes
import math

class Mesh(ResourcesObj):
    _BaseName = 'Mesh'

    def __init__(self, name):
        super().__init__(name)
        currentID = GetOrAddGlobalValue('_MeshCount', 1) # 0 is reserved
        self._meshID = currentID # for corresponding map
        SetGlobalValue('_MeshCount', currentID+1)

    @property
    def meshID(self):
        return self._meshID

    def load(self, path:str):
        '''Load data from file. Override this function to implement loading data from file'''
        raise NotImplementedError
    def draw(self, group:int=None):
        '''
        Draw the mesh. Override this function to implement drawing the mesh.
        :param group: the group of the mesh. Groups can have different materials.
        '''
        raise NotImplementedError

    @classmethod
    def Load(cls, path: str, name=None)->'Mesh':
        '''
        load a mesh from file
        :param path: path of the mesh file
        :param name: name is used to identify the mesh. If not specified, the name will be the path of the mesh file
        :return: Mesh object
        '''
        path, name = cls._GetPathAndName(path, name)
        if '.' not in os.path.basename(path):
            raise ValueError('path must contain file extension')
        ext = path.split('.')[-1]
        formatCls = cls.FindFormat(ext)
        if formatCls is None:
            raise ValueError(f'unsupported mesh format: {ext}')
        mesh = formatCls(name)
        mesh.load(path)
        return mesh

    # region default meshes
    _Plane_Mesh = None
    @classmethod
    def Plane(cls)->'Mesh':
        '''
        get a plane mesh
        '''
        if cls._Plane_Mesh is None:
            cls._Plane_Mesh = _Mesh_Plane()
        return cls._Plane_Mesh
    _Sphere_Meshes = {}
    @classmethod
    def Sphere(cls, segment=32)->'Mesh':
        '''
        get a sphere mesh with specified segment
        :param segment: the segment of the sphere. More segments means more smooth sphere.
        :return: Mesh object
        '''
        if segment not in cls._Sphere_Meshes:
            cls._Sphere_Meshes[segment] = _Mesh_Sphere(segment)
        return cls._Sphere_Meshes[segment]
    _Cube_Mesh = None
    @classmethod
    def Cube(cls)->'Mesh':
        '''
        get a cube mesh with edge length 1
        '''
        if cls._Cube_Mesh is None:
            cls._Cube_Mesh = _Mesh_Cube()
        return cls._Cube_Mesh
    # endregion

# region basic shapes
class _Mesh_Plane(Mesh):
    def __new__(cls):
        return super().__new__(cls, '_Default_Plane_Mesh')
    def __init__(self):
        super().__init__(self.name)
        self.vao = None
        self.vbo = None
        self.vertices = [ # x, y, z, normal_x, normal_y, normal_z, u, v
            -1, 0, -1, 0, 1, 0, 0, 0,
            -1, 0, 1, 0, 1, 0, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 1,
            1, 0, -1, 0, 1, 0, 1, 0
        ]
    def sendToGPU(self):
        if self.vao is not None:
            return  # already sent to GPU
        super().sendToGPU()
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        vertices_data = np.array(self.vertices, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)

        stride = 8 * 4
        # pos
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        # normal
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        # uv
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    def draw(self, group:int=None):
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
    def clear(self):
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(1, [self.vbo])
            self.vao = None
            self.vbo = None
class _Mesh_Sphere(Mesh):
    def __new__(cls, segment:int):
        return super().__new__(cls, f'_Default_Sphere_Mesh_{segment}')
    def __init__(self, segment:int):
        super().__init__(self.name)
        self.segment = segment
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vertices = []
        self._init_vertices(self.segment)
    def _init_vertices(self, segment):
        for i in range(segment+1):
            for j in range(segment+1):
                xSegment = j / segment
                ySegment = i / segment
                xPos = math.cos(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                yPos = math.cos(ySegment * math.pi)
                zPos = math.sin(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                self.vertices.extend([
                    xPos *0.5, yPos*0.5, zPos*0.5,
                    xPos, yPos, zPos,
                    xSegment, ySegment
                ])

        self.indices = []
        oddRow = False
        for i in range(segment):
            if not oddRow:
                for j in range(segment):
                    self.indices.append((i + 1) * (segment + 1) + j)
                    self.indices.append(i * (segment + 1) + j)
            else:
                for j in range(segment, -1, -1):
                    self.indices.append(i * (segment + 1) + j)
                    self.indices.append((i + 1) * (segment + 1) + j)
    def sendToGPU(self):
        if self.vao is not None:
            return
        super().sendToGPU()
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        vertices_data = np.array(self.vertices, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)

        stride = 8 * 4
        # pos
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        # normal
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        # uv
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        indices_data = np.array(self.indices, dtype=np.uint32)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_data.nbytes, indices_data.data, gl.GL_STATIC_DRAW)

    def draw(self, group:int=None):
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLE_STRIP, len(self.indices), gl.GL_UNSIGNED_INT, None)
    def clear(self):
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(1, [self.vbo])
            gl.glDeleteBuffers(1, [self.ebo])
            self.vao = None
            self.vbo = None
            self.ebo = None
class _Mesh_Cube(Mesh):
    def __new__(cls):
        return super().__new__(cls, '_Default_Cube_Mesh')
    def __init__(self):
        super().__init__(self.name)
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vertices = [
            # x, y, z, nx, ny, nz, u, v
            -0.5, -0.5, -0.5, 0, 0, -1, 0, 0,
            0.5, -0.5, -0.5, 0, 0, -1, 1, 0,
            0.5, 0.5, -0.5, 0, 0, -1, 1, 1,
            -0.5, 0.5, -0.5, 0, 0, -1, 0, 1,

            -0.5, -0.5, 0.5, 0, 0, 1, 1, 0,
            0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
            0.5, 0.5, 0.5, 0, 0, 1, 0, 1,
            -0.5, 0.5, 0.5, 0, 0, 1, 1, 1,
        ]
        self.indices = [
            0, 2, 1, 2, 0, 3,
            1, 6, 5, 6, 1, 2,
            7, 5, 6, 5, 7, 4,
            4, 3, 0, 3, 4, 7,
            4, 1, 5, 1, 4, 0,
            3, 6, 2, 6, 3, 7,
        ] # 逆時針

    def sendToGPU(self):
        if self.vao is not None:
            return
        super().sendToGPU()
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        vertices_data = np.array(self.vertices, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)

        stride = 8 * 4
        # pos
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        # normal
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        # uv
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        indices_data = np.array(self.indices, dtype=np.uint32)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_data.nbytes, indices_data.data, gl.GL_STATIC_DRAW)

    def draw(self, group:int=None):
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
    def clear(self):
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(1, [self.vbo])
            gl.glDeleteBuffers(1, [self.ebo])
            self.vao = None
            self.vbo = None
            self.ebo = None
# endregion

__all__ = ['Mesh']