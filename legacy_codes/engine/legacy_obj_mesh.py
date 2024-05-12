import os.path
import ctypes
import OpenGL.GL as gl
import numpy as np
from typing import List, Final
from tqdm import tqdm
from dataclasses import dataclass

from ..source.engine.static.mesh.mesh import Mesh
from engine.static.enums import PrimitiveDrawingType
from common_utils.math_utils import calculate_tangent

@dataclass
class MTL_Data:
    fromFaceIndex:int
    faceCount:int
    materialName:str

class Mesh_OBJ(Mesh):
    '''Specialized Mesh for .obj file format. It can load .obj file and send the data to GPU.'''

    _Format: Final[str] = 'obj'

    _materials: List[MTL_Data]
    '''The material here is not the same as the material in the engine, but MTL_Data'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._materials = []

    @property
    def materials(self):
        '''
        MTL data list. Each element is a MTL_Data object.
        Note: this is not the same as the `material` in the engine.
        '''
        return self._materials

    def clear(self):
        super().clear()
        if self.vbo is not None:
            buffer = np.array([self.vbo], dtype=np.uint32)
            gl.glDeleteBuffers(1, buffer)
            self._vbo = None
        if self.vao is not None:
            buffer = np.array([self.vao], dtype=np.uint32)
            gl.glDeleteVertexArrays(1, buffer)
            self._vao = None
        self._has_normals = None
        self._has_uvs = None
        self.positions.clear()
        self.materials.clear()
        self._vertexCountPerFace = 0
        self._totalFaceCount = 0

    def load(self, path:str):
        temp_points = []
        temp_normals = []
        temp_uvs = []
        faceCount = 0
        lastMatEnd = 0
        currMatName = None
        if not os.path.exists(path):
            raise FileNotFoundError(f'File "{path}" not found')
        with open(path, 'r') as f:
            data_lines = f.readlines()
        mesh_data_lines = [line.strip('\n') for line in data_lines]
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
                face_vertices = line.split(' ')[1:]
                if self.draw_mode is None:
                    if len(face_vertices) == 3:
                        self._drawMode = PrimitiveDrawingType.TRIANGLES
                    elif len(face_vertices) == 4:
                        self._drawMode = PrimitiveDrawingType.QUADS
                    elif len(face_vertices) > 4:
                        self._drawMode = PrimitiveDrawingType.POLYGON
                    else:
                        raise ValueError(f'invalid faces: {line}')
                    self._vertexCountPerFace = len(face_vertices)

                points = []     # [[x1, y1, z1], [x2, y2, z2], ...]
                uvs = []        # [[u1, v1], [u2, v2], ...]
                normals = []    # [[nx1, ny1, nz1], [nx2, ny2, nz2], ...]
                for i, vertex_str in enumerate(face_vertices):
                    vertex = vertex_str.split('/')
                    if len(vertex) == 1: # only point index
                        index = int(vertex[0]) - 1
                        points.append(temp_points[index * 3 : index * 3 + 3])
                    elif (len(vertex)==2): # point index and uv index
                        ptIndex = int(vertex[0]) - 1
                        uvIndex = int(vertex[1]) - 1
                        points.append(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                        uvs.append(temp_uvs[uvIndex * 2 : uvIndex * 2 + 2])
                    elif len(vertex)==3:
                        if vertex[1] == '': # point index and normal index
                            ptIndex = int(vertex[0]) - 1
                            normalIndex = int(vertex[2]) - 1
                            points.append(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                            normals.append(temp_normals[normalIndex * 3 : normalIndex * 3 + 3])
                        else: # point index, uv index and normal index
                            ptIndex = int(vertex[0]) - 1
                            uvIndex = int(vertex[1]) - 1
                            normalIndex = int(vertex[2]) - 1
                            points.append(temp_points[ptIndex * 3 : ptIndex * 3 + 3])
                            normals.append(temp_normals[normalIndex * 3: normalIndex * 3 + 3])
                            uvs.append(temp_uvs[uvIndex * 2 : uvIndex * 2 + 2])
                    else:
                        raise ValueError(f'invalid faces: {line}')

                    if i == len(face_vertices) - 1:     # append data to vertices list
                        for k in range(len(points)):
                            self.positions.extend(points[k])
                            if len(temp_normals) > 0:
                                self.positions.extend(normals[k])
                            if len(temp_uvs) > 0:
                                self.positions.extend(uvs[k])
                            if len(temp_uvs)>0 and len(temp_normals)>0:
                                # calculate tangent and bitangent for this vertex
                                center_point = points[k]
                                left_point = points[k-1 if k>0 else len(points)-1]
                                right_point = points[k+1 if k<len(points)-1 else 0]
                                center_uv = uvs[k]
                                left_uv = uvs[k-1 if k>0 else len(uvs)-1]
                                right_uv = uvs[k+1 if k<len(uvs)-1 else 0]
                                tangent = calculate_tangent(np.array(center_point, dtype=np.float32),
                                                            np.array(left_point, dtype=np.float32),
                                                            np.array(right_point, dtype=np.float32),
                                                            np.array(center_uv, dtype=np.float32),
                                                            np.array(left_uv, dtype=np.float32),
                                                            np.array(right_uv, dtype=np.float32))
                                self.positions.extend(tangent.tolist())
                faceCount += 1
        if currMatName is not None:
            self.materials.append(MTL_Data(lastMatEnd, faceCount - lastMatEnd, currMatName))
        self._has_normals = len(temp_normals) > 0
        self._has_uvs = len(temp_uvs) > 0
        self._totalFaceCount = faceCount

    def load(self):
        if self.vao is not None:
            return # already sent to GPU

        super().load()

        if len(self.positions) == 0:
            raise Exception(f'Mesh `{self.name}` has no data to send to GPU')
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        vertices_data = np.array(self.positions, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)

        if self.has_normals and self.has_uvs:
            stride = (3 + 3 + 2 + 3) * 4    # position + normal + uv + tangent
        elif self.has_normals:
            stride = (3 + 3) * 4    # position + normal
        elif self.has_uvs:
            stride = (3 + 2) * 4    # position + uv
        else:
            stride = 3 * 4   # position only

        # position
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        # normal
        if self.has_normals: # if there is no normal data, then no need to send
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        # uv
        if self.has_uvs: # if there is no uv data, then no need to send
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p((3 + self.has_normals * 3) * 4))
        # tangent
        if self.has_normals and self.has_uvs:
            gl.glEnableVertexAttribArray(3)
            gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p((3 + 3 + 2) * 4))

        if not self.keep_vertices:
            self.positions.clear()

    def draw(self, group:int=None):
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)

        if len(self.materials) == 0 or group is None: # no material
            gl.glDrawArrays(self.draw_mode.value, 0, self.totalFaceCount * self.vertexCountPerFace)
        else:
            if group >= len(self.materials):
                return # incorrect material slot
            mat = self.materials[group]
            gl.glDrawArrays(self.draw_mode.value, mat.fromFaceIndex * self.vertexCountPerFace, mat.faceCount * self.vertexCountPerFace)



__all__ = ['Mesh_OBJ', 'MTL_Data']