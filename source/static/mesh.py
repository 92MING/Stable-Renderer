from static.enums import *
from utils.global_utils import GetOrAddGlobalValue
from utils.base_clses.named_obj import NamedObj
import OpenGL.GL as gl
import numpy as np
import ctypes

_ALL_MESH_TYPES:dict = GetOrAddGlobalValue('_ALL_MESH_TYPES', dict())  # e.g. 'obj': MeshData_Obj. Note that only one MeshData class can be registered for each format
def suppportedMeshFormats():
    return tuple(_ALL_MESH_TYPES.keys())

class _MeshDataMetaCls(type):
    def __new__(self, *args, **kwargs):
        cls_name = args[0]
        for cur_cls in _ALL_MESH_TYPES.values():
            if cur_cls.__qualname__ == cls_name:
                return cur_cls
        cls = super().__new__(self, *args, **kwargs)
        if cls.__qualname__ == 'MeshData':
            return cls
        format = cls.format
        if format in _ALL_MESH_TYPES:
            raise Exception(f'MeshData Class with format {format} already exists')
        _ALL_MESH_TYPES[format] = cls
        return cls
class MeshData(metaclass=_MeshDataMetaCls):
    format: str = None
    '''override this property to specify the format of the mesh data, e.g. "obj"'''
    def __class_getitem__(cls, item):
        if item.startswith('.'):
            item = item[1:]
        item = item.lower()
        if item in _ALL_MESH_TYPES:
            return _ALL_MESH_TYPES[item]
        else:
            raise ValueError(f'MeshData Class with format {item} does not exist')
    def load(self, path:str):
        '''Load data from file. Override this function to implement loading data from file'''
        raise NotImplementedError
    def sendToGPU(self):
        '''Send data to GPU memory'''
        raise NotImplementedError
    def clear(self):
        '''Clear all data and release GPU memory if it has been loaded'''
        raise NotImplementedError
    def draw(self):
        '''Draw the mesh'''
        raise NotImplementedError
class MeshData_OBJ(MeshData):

    format:str = 'obj'

    def __init__(self):
        super().__init__()
        self.drawMode : PrimitiveType = None
        self.vao = None
        self.vbo = None
        self.has_normals = False
        self.has_uvs = False
        self.vertexCount = 0 # number of vertices per face
        self.vertices = []
    def clear(self):
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
        self.clear()
        temp_points = []
        temp_normals = []
        temp_uvs = []
        for line in open(path, 'r').readlines():
            if line.startswith('#'): continue
            if line.startswith('v'): temp_points.extend(map(float, line.split(' ')[1:]))
            if line.startswith('vn'): temp_normals.extend(map(float, line.split(' ')[1:]))
            if line.startswith('vt'): temp_uvs.extend(map(float, line.split(' ')[1:]))
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
                for face in faces:
                    face = face.split('/')
                    if len(face) == 1: # only point index
                        self.vertices.append(temp_points[int(face[0]) - 1])
                    elif (len(face)==2): # point index and uv index
                        self.vertices.append(temp_points[int(face[0]) - 1])
                        self.vertices.append(temp_uvs[int(face[1]) - 1])
                    elif len(face)==3:
                        if face[1] == '': # point index and normal index
                            self.vertices.append(temp_points[int(face[0]) - 1])
                            self.vertices.append(temp_normals[int(face[2]) - 1])
                        else: # point index, uv index and normal index
                            self.vertices.append(temp_points[int(face[0]) - 1])
                            self.vertices.append(temp_uvs[int(face[1]) - 1])
                            self.vertices.append(temp_normals[int(face[2]) - 1])
                    else:
                        raise ValueError(f'invalid faces: {line}')
                    self.vertexCount = len(face)
            self.has_normals = len(temp_normals) > 0
            self.has_uvs = len(temp_uvs) > 0
    def sendToGPU(self):
        if self.vao is not None:
            return # already sent to GPU
        if len(self.vertices) == 0:
            raise Exception('No data to send to GPU')
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        vertices_data = np.array(self.vertices, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data, gl.GL_STATIC_DRAW)

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

        # normal
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
        gl.glDrawArrays(self.drawMode.value, 0, self.vertexCount)

class Mesh(NamedObj):

    def __init__(self, name, data:MeshData=None):
        super().__init__(name)
        self._meshData :MeshData = data
        self._format = None # obj, stl, etc.

    # region properties
    @property
    def drawMode(self):
        '''draw mode of the mesh, e.g. gl.GL_TRIANGLES, gl.GL_QUADS, etc.'''
        if self._meshData is None:
            raise ValueError('MeshData is not loaded')
        return self._meshData.drawMode
    @property
    def format(self):
        '''file format of the mesh, e.g. obj, stl, etc.'''
        if self._meshData is None:
            raise ValueError('MeshData is not loaded')
        return self._meshData.format
    # endregion

    def sendToGPU(self):
        if self._meshData is None:
            raise ValueError('MeshData is not loaded')
        self._meshData.sendToGPU()

    @classmethod
    def Load(cls, path:str, name=None):
        '''
        load a mesh from file
        :param path: path of the mesh file
        :param name: name is used to identify the mesh. If not specified, the name will be the path of the mesh file
        :return: Mesh object
        '''
        if name is None:
            name = path
        if '.' not in name:
            raise ValueError('name must contain file extension')
        ext = name.split('.')[-1]
        data = MeshData[ext]()
        data.load(path)
        mesh = Mesh(name, data)
        return mesh

if __name__ == '__main__':
    mesh = Mesh.Load('../../resources/boat/boat.obj')
