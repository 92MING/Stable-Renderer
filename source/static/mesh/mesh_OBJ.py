from static.enums import PrimitiveType
import OpenGL.GL as gl
import numpy as np
import ctypes
from .mesh import Mesh

class Mesh_OBJ(Mesh):

    _Format = 'obj'

    def __init__(self, name):
        super().__init__(name)
        self.drawMode : PrimitiveType = None
        self.vao = None
        self.vbo = None
        self.has_normals = False
        self.has_uvs = False
        self.vertexCount = 0 # number of vertices per face
        self.vertices = []
    def clear(self):
        super().clear()
        if self.vbo is not None:
            gl.glDeleteBuffers(1, self.vbo)
            self.vbo = None
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, self.vao)
            self.vao = None
        self.has_normals = False
        self.has_uvs = False
        self.vertices.clear()
    def load(self, path:str):
        temp_points = []
        temp_normals = []
        temp_uvs = []
        for line in open(path, 'r').readlines():
            if line.startswith('#'): continue
            if line.startswith('v'): temp_points.extend(map(float, line.split(' ')[1:4]))
            if line.startswith('vn'): temp_normals.extend(map(float, line.split(' ')[1:4]))
            if line.startswith('vt'): temp_uvs.extend(map(float, line.split(' ')[1:3]))
            if line.startswith('f'):
                faces = line.split(' ')[1:]
                if self.drawMode is None:
                    if len(faces) == 3:
                        self.drawMode = PrimitiveType.TRIANGLES
                    elif len(faces) == 4:
                        self.drawMode = PrimitiveType.QUADS
                    elif len(faces) > 4:
                        self.drawMode = PrimitiveType.POLYGON
                    else:
                        raise ValueError(f'invalid faces: {line}')
                    self.vertexCount = len(faces)
                for face in faces:
                    face = face.split('/')
                    if len(face) == 1: # only point index
                        index = int(face[0]) - 1
                        self.vertices.extend(temp_points[index * 3 : index * 3 + 3])
                    elif (len(face)==2): # point index and uv index
                        ptIndex = int(face[0]) - 1
                        uvIndex = int(face[1]) - 1
                        self.vertices.extend(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                        self.vertices.extend(temp_uvs[uvIndex * 2 : uvIndex * 2 + 2])
                    elif len(face)==3:
                        if face[1] == '': # point index and normal index
                            ptIndex = int(face[0]) - 1
                            normalIndex = int(face[2]) - 1
                            self.vertices.extend(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                            self.vertices.extend(temp_normals[normalIndex * 3 : normalIndex * 3 + 3])
                        else: # point index, uv index and normal index
                            ptIndex = int(face[0]) - 1
                            uvIndex = int(face[1]) - 1
                            normalIndex = int(face[2]) - 1
                            self.vertices.extend(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                            self.vertices.extend(temp_normals[normalIndex * 3: normalIndex * 3 + 3])
                            self.vertices.extend(temp_uvs[uvIndex * 2 : uvIndex * 2 + 2])
                    else:
                        raise ValueError(f'invalid faces: {line}')
        self.has_normals = len(temp_normals) > 0
        self.has_uvs = len(temp_uvs) > 0
    def sendToGPU(self):
        super().sendToGPU()
        if self.vao is not None:
            return # already sent to GPU
        if len(self.vertices) == 0:
            raise Exception('No data to send to GPU')
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        vertices_data = np.array(self.vertices, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)

        if self.has_normals and self.has_uvs:
            stride = 8 * 4
        elif self.has_normals:
            stride = 6 * 4
        elif self.has_uvs:
            stride = 5 * 4
        else:
            stride = 3 * 4

        # position
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)

        # normal6
        if self.has_normals: # if there is no normal data, then no need to send
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        # uv
        if self.has_uvs: # if there is no uv data, then no need to send
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p((3 + self.has_normals * 3) * 4))

    def draw(self):
        if len(self.vertices) == 0:
            raise Exception('No data to draw')
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(self.drawMode.value, 0, len(self.vertices) // self.vertexCount)

__all__ = ['Mesh_OBJ']