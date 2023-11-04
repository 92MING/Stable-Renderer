import os.path

from static.enums import PrimitiveType
import OpenGL.GL as gl
import numpy as np
import ctypes
from .mesh import Mesh
from typing import List
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class MTL_Data:
    fromFaceIndex:int
    faceConut:int
    materialName:str

class Mesh_OBJ(Mesh):

    _Format = 'obj'

    def __init__(self, name):
        super().__init__(name)
        self.drawMode : PrimitiveType = None
        self.vao = None
        self.vbo = None
        self.has_normals = False
        self.has_uvs = False
        self.vertexCountPerFace = 0 # number of vertices per face
        self.totalFaceCount = 0
        self.vertices = []
        self.materials:List[MTL_Data] = []
        '''The material here is not the same as the material in the engine, but MTL_Data'''

    def clear(self):
        super().clear()
        if self.vbo is not None:
            buffer = np.array([self.vbo], dtype=np.uint32)
            gl.glDeleteBuffers(1, buffer)
            self.vbo = None
        if self.vao is not None:
            buffer = np.array([self.vao], dtype=np.uint32)
            gl.glDeleteVertexArrays(1, buffer)
            self.vao = None
        self.has_normals = False
        self.has_uvs = False
        self.vertices.clear()
        self.materials.clear()
        self.vertexCountPerFace = 0
        self.totalFaceCount = 0

    def load(self, path:str):
        temp_points = []
        temp_normals = []
        temp_uvs = []
        faceCount = 0
        lastMatEnd = 0
        currMatName = None
        if not os.path.exists(path):
            raise FileNotFoundError(f'File "{path}" not found')
        mesh_data_lines = [line.strip('\n') for line in open(path, 'r').readlines()]
        for line in tqdm(mesh_data_lines, desc=f'Loading Mesh-"{self.name}"'):
            if line.startswith('#'): continue
            if line.startswith('v'): temp_points.extend(map(float, line.split(' ')[1:4]))
            if line.startswith('vn'): temp_normals.extend(map(float, line.split(' ')[1:4]))
            if line.startswith('vt'): temp_uvs.extend(map(float, line.split(' ')[1:3]))
            if line.startswith('usemtl'):
                if currMatName is None:
                    currMatName = line.split(' ')[1]
                else:
                    neMatName = line.split(' ')[1]
                    self.materials.append(MTL_Data(lastMatEnd, faceCount - lastMatEnd, currMatName))
                    lastMatEnd = faceCount
                    currMatName = neMatName
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
                    self.vertexCountPerFace = len(faces)
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
                faceCount += 1
        if currMatName is not None:
            self.materials.append(MTL_Data(lastMatEnd, faceCount - lastMatEnd, currMatName))
        self.has_normals = len(temp_normals) > 0
        self.has_uvs = len(temp_uvs) > 0
        self.totalFaceCount = faceCount

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

    def draw(self, slot:int=None):
        '''Material.use() must be called before calling this function'''
        if len(self.vertices) == 0:
            raise Exception('No data to draw')
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)

        if len(self.materials) == 0 or slot is None: # no material
            gl.glDrawArrays(self.drawMode.value, 0, self.totalFaceCount * self.vertexCountPerFace)
        else:
            if slot >= len(self.materials):
                return # incorrect material slot
            mat = self.materials[slot]
            gl.glDrawArrays(self.drawMode.value, mat.fromFaceIndex * self.vertexCountPerFace, mat.faceConut * self.vertexCountPerFace)



__all__ = ['Mesh_OBJ', 'MTL_Data']