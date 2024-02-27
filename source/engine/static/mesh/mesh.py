import os
import OpenGL.GL as gl
import numpy as np
import ctypes
import math
from typing import List, Final, Union, Type

from utils.global_utils import GetOrAddGlobalValue, SetGlobalValue
from engine.static.resourcesObj import ResourcesObj
from engine.static.enums import PrimitiveType


class Mesh(ResourcesObj):
    '''Mesh class. It is used to store the vertices, normals, uvs, etc. of a mesh. It also provides functions to load and draw the mesh.'''

    # region class properties
    _BaseName: Final[str] = 'Mesh'
    '''Basename is for internal use only. All subclass of Mesh will have the same basename. Do not change it.'''
    # endregion

    # region instance properties
    _meshID: int
    '''MeshID is a unique ID for each mesh. It is used to identify the mesh. It is automatically generated when the mesh is created.'''

    # AI stuff
    _generateVertexID: bool = False
    '''If True, vertex IDs will be generated when loading the mesh. This is a technique for AI rendering.'''
    _vertexBoundingBoxLength: float = 0.0
    '''
    The length of the bounding box when generating vertex IDs. This is a technique for AI rendering.
    This will only be used when `generateVertexID` = True. 
    When:    
        * `vertexBoundingBoxLength` = 0: means every vertex has its own ID.
        * `vertexBoundingBoxLength` > 0: means vertices within the same bounding box will have the same ID. This is a technique for AI rendering.
    '''

    # mesh data
    _vertices: List[float]
    '''
    vertices data.
    Format:
        * has normal & uv: [x, y, z, nx, ny, nz, u, v, ...]
        * has normal only: [x, y, z, nx, ny, nz, ...]
        * has uv only: [x, y, z, u, v, ...]
        * has neither normal nor uv: [x, y, z, ...]
    '''
    _vertexCountPerFace: int = 0
    '''number of vertices per face'''
    _totalFaceCount: int = 0
    '''total number of faces'''
    _has_normals: Union[bool, None] = None
    '''If the mesh has normals. `None` means unknown'''
    _has_uvs: Union[bool, None] = None
    '''If the mesh has uvs. `None` means unknown'''

    # opengl stuff
    _vao: Union[int, None] = None
    '''Vertex Array Object. `None` means not sent to GPU. Otherwise, it is the buffer ID.'''
    _vbo: Union[int, None] = None
    '''Vertex Buffer Object. `None` means not sent to GPU. Otherwise, it is the buffer ID.'''
    _drawMode: Union[PrimitiveType, None] = None
    '''The OpenGL drawing mode of the mesh. `None` means unknown. It should be set when `load` is called'''
    _keep_vertices: bool = False
    '''If keep vertices, the vertices data will be kept in memory(after send to GPU). Otherwise, they will be cleared.'''
    # endregion

    @staticmethod
    def _GenerateMeshID()->int:
        currentID = GetOrAddGlobalValue('_MeshCount', 1) # 0 is reserved
        SetGlobalValue('_MeshCount', currentID+1)
        return currentID

    def __init__(self,
                 name: str,
                 keep_vertices:bool=False,
                 generateVertexID:bool=False,
                 vertexBoundingBoxLength:float=0.0):
        '''
        Args:
            * name: name is used to identify the mesh. If not specified, the name will be the path/ auto-generated.
            * keep_vertices: if True, vertex data will still be kept after sent to GPU.
            * generateVertexID: if True, vertex IDs will be generated when loading the mesh. This is a technique for AI rendering.
            * vertexBoundingBoxLength: The length of the bounding box when generating vertex IDs. This is a technique for AI rendering.
                                        This will only be used when `generateVertexID` = True.
                                        When:
                                            * `vertexBoundingBoxLength` = 0: means every vertex has its own ID.
                                            * `vertexBoundingBoxLength` > 0: means vertices within the same bounding box will have the same ID. This is a technique for AI rendering.
        '''
        super().__init__(name)
        self._meshID = self._GenerateMeshID()
        self._vertices = []
        self._keep_vertices = keep_vertices
        self._generateVertexID = generateVertexID
        if vertexBoundingBoxLength < 0:
            raise ValueError('vertexBoundingBoxLength must be non-negative')
        self._vertexBoundingBoxLength = vertexBoundingBoxLength

    # region properties
    @property
    def meshID(self):
        '''MeshID is a unique ID for each mesh. It is used to identify the mesh. It is automatically generated when the mesh is created.'''
        return self._meshID
    
    @property
    def vertexCountPerFace(self):
        '''number of vertices per face'''
        return self._vertexCountPerFace
    
    @property
    def totalFaceCount(self):
        return self._totalFaceCount
    
    @property
    def vao(self):
        return self._vao
    
    @property
    def vbo(self):
        return self._vbo

    @property
    def sentToGPU(self):
        '''Check if the mesh data is sent to GPU.'''
        return self.vao is not None
    
    @property
    def keep_vertices(self):
        '''If keep vertices, the vertices data will be kept in memory(after send to GPU). Otherwise, they will be cleared.'''
        return self._keep_vertices

    @property
    def generateVertexID(self):
        '''If True, vertex IDs will be generated when loading the mesh. This is a technique for AI rendering.'''
        return self._generateVertexID

    @generateVertexID.setter
    def generateVertexID(self, value:bool):
        if self.sentToGPU:
            raise Exception('Cannot change generateVertexID after data is sent to GPU')
        self._generateVertexID = value

    @property
    def vertexBoundingBoxLength(self):
        '''
        The length of the bounding box when generating vertex IDs. This is a technique for AI rendering.
        This will only be used when `generateVertexID` = True.
        When:
            * `vertexBoundingBoxLength` = 0: means every vertex has its own ID.
            * `vertexBoundingBoxLength` > 0: means vertices within the same bounding box will have the same ID. This is a technique for AI rendering.
        '''
        return self._vertexBoundingBoxLength

    @vertexBoundingBoxLength.setter
    def vertexBoundingBoxLength(self, value:float):
        if self.sentToGPU:
            raise Exception('Cannot change vertexBoundingBoxLength after data is sent to GPU')
        if value < 0:
            raise ValueError('vertexBoundingBoxLength must be non-negative')
        self._vertexBoundingBoxLength = value

    @property
    def has_normals(self):
        if self._has_normals is None:
            raise Exception('has_normals is unknown. Please load the mesh first.')
        return self._has_normals
    
    @property
    def has_uvs(self):
        if self._has_uvs is None:
            raise Exception('has_uvs is unknown. Please load the mesh first.')
        return self._has_uvs
    
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def drawMode(self):
        return self._drawMode
    # endregion

    # region methods
    def load(self, path:str):
        '''Load data from file. Override this function to implement loading data from file'''
        raise NotImplementedError

    def draw(self, group:int=None):
        '''
        Draw the mesh. Override this function to implement drawing the mesh.
        Make sure you have called Material.use() before calling this function. (So that textures & shaders are ready)
        :param group: the group of the mesh. Groups can have different materials.
        '''
        # default implementation
        if self.vao is None:
            raise Exception('Data is not sent to GPU')
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(self.drawMode.value, 0, self.totalFaceCount * self.vertexCountPerFace)

    # endregion

    @classmethod
    def Load(cls,
             path: str,
             name: Union[str, None]=None,
             keep_vertices:bool=False,
             generateVertexID:bool=False,
             vertexBoundingBoxLength:float=0.0,
             )->'Mesh':
        '''
        load a mesh from file

        Args:
            * path: path of the mesh file
            * name: name is used to identify the mesh. If not specified, the name will be the path of the mesh file
            * keep_vertices: if True, the vertices data will be kept in memory(after send to GPU). Otherwise, they will be cleared.
            * generateVertexID: if True, vertex IDs will be generated when loading the mesh. This is a technique for AI rendering.
            * vertexBoundingBoxLength: The length of the bounding box when generating vertex IDs. This is a technique for AI rendering.
                                        This will only be used when `generateVertexID` = True.
                                        When:
                                            * `vertexBoundingBoxLength` = 0: means every vertex has its own ID.
                                            * `vertexBoundingBoxLength` > 0: means vertices within the same bounding box will have the same ID. This is a technique for AI rendering.
        '''
        path, name = cls._GetPathAndName(path, name)
        if '.' not in os.path.basename(path):
            raise ValueError('path must contain file extension')
        ext = path.split('.')[-1]
        formatCls:Type['Mesh'] = cls.FindFormat(ext)
        if formatCls is None:
            raise ValueError(f'unsupported mesh format: {ext}')
        mesh = formatCls(name=name,
                         keep_vertices=keep_vertices,
                         generateVertexID=generateVertexID,
                         vertexBoundingBoxLength=vertexBoundingBoxLength)
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

    _plane_vertices = None
    '''For plane mesh, the vertices data is the same. So we can use the same data for all plane meshes.'''

    @classmethod
    def _GetPlaneVertices(cls):
        if cls._plane_vertices is None:
            cls._plane_vertices = [
                -1, 0, -1, 0, 1, 0, 0, 0,
                -1, 0, 1, 0, 1, 0, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 1,
                1, 0, -1, 0, 1, 0, 1, 0
            ]
        return cls._plane_vertices

    def __new__(cls):
        return super().__new__(cls, '_Default_Plane_Mesh')

    def __init__(self):
        super().__init__(self.name, keep_vertices=True)
        self._vao = None
        self._vbo = None
        self._drawMode = PrimitiveType.TRIANGLE_FAN
        self._vertexCountPerFace = 4
        self._totalFaceCount = 1
        self._has_normals = True
        self._has_uvs = True

    @property
    def vertices(self):
        return self._GetPlaneVertices()

    def sendToGPU(self):
        if self.vao is not None:
            return  # already sent to GPU
        super().sendToGPU()
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self._vbo = gl.glGenBuffers(1)
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
            self._vao = None
            self._vbo = None

class _Mesh_Sphere(Mesh):

    def __new__(cls, segment:int):
        return super().__new__(cls, f'_Default_Sphere_Mesh_{segment}')

    def __init__(self, segment:int):
        super().__init__(self.name)
        self.segment = segment
        self._ebo = None
        self._vertexCountPerFace = 3
        self._drawMode = PrimitiveType.TRIANGLE_STRIP
        self._init_vertices(self.segment)
        self._has_normals = True
        self._has_uvs = True
    @property
    def ebo(self):
        return self._ebo
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
        for i in range(segment):
            for j in range(segment):
                self.indices.append((i + 1) * (segment + 1) + j)
                self.indices.append(i * (segment + 1) + j)
        self._totalFaceCount = len(self.indices) - 2

    def sendToGPU(self):
        if self.vao is not None:
            return
        super().sendToGPU()
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self._vbo = gl.glGenBuffers(1)
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

        self._ebo = gl.glGenBuffers(1)
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
            self._vao = None
            self._vbo = None
            self._ebo = None

class _Mesh_Cube(Mesh):

    _cube_vertices:List[float] = None
    _cube_indices:List[int] = None

    @classmethod
    def _GetCubeVertices(cls)->List[float]:
        if cls._cube_vertices is None:
            cls._cube_vertices = [
                # x, y, z,         nx, ny, nz,   u, v
                -0.5, -0.5, -0.5,  0, 0, -1,    0, 0,
                0.5, -0.5, -0.5,   0, 0,  1,    1, 0,
                0.5, 0.5, -0.5,    0, 0, -1,    1, 1,
                -0.5, 0.5, -0.5,   0, 0, -1,    0, 1,

                -0.5, -0.5, 0.5,   0, 0, 1,     1, 0,
                0.5, -0.5, 0.5,    0, 0, 1,     0, 0,
                0.5, 0.5, 0.5,     0, 0, 1,     0, 1,
                -0.5, 0.5, 0.5,    0, 0, 1,     1, 1,
            ]
        return cls._cube_vertices

    @classmethod
    def _GetCubeIndices(cls)->List[int]:
        if cls._cube_indices is None:
            cls._cube_indices = [
                0, 2, 1, 2, 0, 3,
                1, 6, 5, 6, 1, 2,
                7, 5, 6, 5, 7, 4,
                4, 3, 0, 3, 4, 7,
                4, 1, 5, 1, 4, 0,
                3, 6, 2, 6, 3, 7,
            ]
        return cls._cube_indices

    def __new__(cls):
        return super().__new__(cls, '_Default_Cube_Mesh')

    def __init__(self):
        super().__init__(self.name, keep_vertices=True)
        self._drawMode = PrimitiveType.TRIANGLES
        self._totalFaceCount = 12
        self._vertexCountPerFace = 3
        self._ebo = None
        self._has_normals = True
        self._has_uvs = True

    @property
    def vertices(self)->List[float]:
        return self._GetCubeVertices()

    @property
    def indices(self)->List[int]:
        return self._GetCubeIndices()

    @property
    def ebo(self):
        return self._ebo

    def sendToGPU(self):
        if self.vao is not None:
            return
        super().sendToGPU()
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self._vbo = gl.glGenBuffers(1)
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

        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
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
            self._vao = None
            self._vbo = None
            self._ebo = None

# endregion

__all__ = ['Mesh']