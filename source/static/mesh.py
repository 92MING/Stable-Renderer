import os.path
from static.enums import *
from utils.global_utils import GetOrAddGlobalValue
import OpenGL.GL as gl

_ALL_MESH_TYPES:dict = GetOrAddGlobalValue('_ALL_MESH_TYPES', dict())  # e.g. 'obj': MeshData_Obj. Note that only one MeshData class can be registered for each format
_ALL_MESHES:dict = GetOrAddGlobalValue('_ALL_MESHES', dict()) # e.g. 'boat': ...
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
    def __init__(self):
        self._drawMode:PrimitiveType = None
    def __class_getitem__(cls, item):
        if item.startswith('.'):
            item = item[1:]
        item = item.lower()
        if item in _ALL_MESH_TYPES:
            return _ALL_MESH_TYPES[item]
        else:
            raise ValueError(f'MeshData Class with format {item} does not exist')

    @property
    def drawMode(self)->PrimitiveType:
        return self._drawMode
    @property
    def format(self)->str:
        '''Return the format of the mesh data, e.g. "obj" '''
        raise NotImplementedError
    def load(self, path:str):
        '''Load data from file. Override this function to implement loading data from file'''
        raise NotImplementedError
    def sendToGPU(self):
        '''Send data to GPU memory'''
        raise NotImplementedError
    def clear(self):
        '''Clear all data and release GPU memory if it has been loaded'''
        raise NotImplementedError
class MeshData_OBJ(MeshData):
    def __init__(self):
        super().__init__()
        self._vao = None
        self._point_vbo = None
        self._normal_vbo = None
        self._uv_vbo = None
        self._ebo = None
        self._points:list = []  # [x1, y1, z1,   x2, y2, z2,   ...]
        self._normals:list = []  # [x1, y1, z1,   x2, y2, z2,   ...]
        self._uvs:list = []  # [u1, v1,   u2, v2,   ...]
        self._indices:list = [] # [i1, i2, ...]

    @property
    def format(self)->str:
        return 'obj'

    def clear(self):
        if self._point_vbo is not None:
            gl.glDeleteBuffers(1, self._point_vbo)
            self._point_vbo = None
        if self._normal_vbo is not None:
            gl.glDeleteBuffers(1, self._normal_vbo)
            self._normal_vbo = None
        if self._uv_vbo is not None:
            gl.glDeleteBuffers(1, self._uv_vbo)
            self._uv_vbo = None
        if self._ebo is not None:
            gl.glDeleteBuffers(1, self._ebo)
            self._ebo = None
        if self._vao is not None:
            gl.glDeleteVertexArrays(1, self._vao)
            self._vao = None
        self._points.clear()
        self._normals.clear()
        self._uvs.clear()
        self._indices.clear()
    def load(self, path:str):
        self.clear()
        for line in open(path, 'r').readlines():
            if line.startswith('#'):
                continue
            if line.startswith('v'):
                pt = line.split(' ')[1:]
                self._points.extend(map(float, pt))
            if line.startswith('vn'):
                normal = line.split(' ')[1:]
                self._normals.extend(map(float, normal))
            if line.startswith('vt'):
                uv = line.split(' ')[1:]
                self._uvs.extend(map(float, uv))
            if line.startswith('f'):
                faces = line.split(' ')[1:]
                if self._drawMode is None:
                    if len(faces) == 3:
                        self._drawMode = PrimitiveType.TRIANGLES
                    elif len(faces) == 4:
                        self._drawMode = PrimitiveType.QUADS
                    elif len(faces) > 4:
                        self._drawMode = PrimitiveType.POLYGON
                    else:
                        raise ValueError(f'invalid faces: {line}')
                    self._facePointCount = len(faces)
                for face in faces:
                    face = face.split('/')
                    if len(face) == 1:
                        self._indices.append((int(face[0]), None, None))
                    elif (len(face)==2) or (len(face)==3 and face[1]==''):
                        self._indices.append((int(face[0]), None, int(face[1])))
                    elif len(face) == 3:
                        self._indices.append((int(face[0]), int(face[1]), int(face[2])))
                    else:
                        raise ValueError(f'invalid faces: {line}')

    def sendToGPU(self):
        if self._vao is not None:
            return # already sent to GPU
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._point_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._point_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(self._points) * self._points[0].size, self._points, gl.GL_STATIC_DRAW)

        if len(self._normals)>0: # if there is no normal data, then no need to send
            self._normal_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._normal_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(self._normals) * self._normals[0].size, self._normals, gl.GL_STATIC_DRAW)

        if len(self._uvs)>0: # if there is no uv data, then no need to send
            self._uv_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._uv_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(self._uvs) * self._uvs[0].size, self._uvs, gl.GL_STATIC_DRAW)

        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(self._vertices) * 3 * 4, self._vertices, gl.GL_STATIC_DRAW)

class Mesh:

    def __new__(cls, name):
        if name in _ALL_MESHES:
            return _ALL_MESHES[name]
        else:
            return super().__new__(cls)
    def __init__(self, name):
        self._name = name
        self._meshData :MeshData = None
        self._drawMode = None
        self._format = None # obj, stl, etc.

    # region properties
    @property
    def name(self):
        '''name is used to identify the mesh. If not specified, the name will be the path of the mesh file'''
        return self._name
    @property
    def drawMode(self):
        '''draw mode of the mesh, e.g. gl.GL_TRIANGLES, gl.GL_QUADS, etc.'''
        if self._drawMode is None:
            raise ValueError('MeshData is not loaded')
        return self._drawMode
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
        if not path.endswith(_SUPPORTED_FORMAT):
            raise ValueError(f'unsupported format: {os.path.basename(path)}. Currently only support {str(_SUPPORTED_FORMAT)[1:-1]}')
        mesh = Mesh(name)
        if path.endswith('.obj'):
            mesh._loadObj(path)
            mesh._format = 'obj'
        _ALL_MESHES[name] = mesh
        return mesh

if __name__ == '__main__':
    mesh = Mesh.Load('../../resources/boat/boat.obj')
