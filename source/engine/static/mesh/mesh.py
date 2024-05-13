import os
import OpenGL.GL as gl
import numpy as np
import ctypes
import math
import assimp_py

from pathlib import Path
from attr import attrs, attrib
from typing import List, Final, Union, Type, Tuple, Optional
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue, is_dev_mode
from engine.static.resources_obj import ResourcesObj
from engine.static.enums import PrimitiveDrawingType

def _get_mesh_id():
    mesh_id = GetOrAddGlobalValue('__ENGINE_MESH_COUNT_ID__', 0)
    SetGlobalValue('__ENGINE_MESH_COUNT_ID__', mesh_id + 1) # type: ignore
    return mesh_id
    
@attrs(eq=False, repr=False)
class InnerMesh:
    '''the inner target inside a loaded mesh data.'''
    
    from_vertex: int = attrib(default=0)
    '''from which vertex inside the whole vertex data list (<x,y,z> as 1 vertex, not <x>, <y>, <z>)'''
    vertex_count: int = attrib(default=0)
    '''how many vertex does this inner target has, (x,y,z) as 1 vertex. If `indices_count` is specified, this could be ignore.'''
    
    from_index: int = attrib(default=0)
    '''from which index inside the whole index data list. If no indices, this field will be ignored.'''
    indices_count: int = attrib(default=0)
    '''how many indices does this inner target has. If this is specified, `vertex_count` will be ignored.'''
    
    material_index: Optional[int] = attrib(default=None)
    '''which material does this inner target use. If None, use the default material.'''
    material_name: Optional[str] = attrib(default=None)
    '''name of this inner target's material. Not all formats have this field.'''
    
    @property
    def to_vertex(self):
        return self.from_vertex + self.vertex_count
    @property
    def to_index(self):
        return self.from_index + self.indices_count
    
@attrs(eq=False, repr=False)
class Mesh(ResourcesObj):
    '''
    Mesh class. It is used to store the vertices, normals, uvs, etc. of a mesh. It also provides functions to load and draw the mesh.
    Note that mesh can have several inner objs, if you have given the inner data, you have to specify the drawing target on `draw`.
    '''

    BaseClsName = 'Mesh'
    '''Basename is for internal use only. All subclass of Mesh will have the same basename. Do not change it.'''

    positions: List[Tuple[float, float, float]] = attrib(factory=list, kw_only=True)
    '''position data, [(x1, y1, z1), ...]'''
    position_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of positions after data is cleared by `only_keep_gpu_data`'''
    
    normals: List[Tuple[float, float, float]] = attrib(factory=list, kw_only=True)
    '''normals data, [(x1, y1, z1), ...]'''
    normal_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of normals after data is cleared by `only_keep_gpu_data`'''
    
    uvs: List[List[Tuple[float, float]]] = attrib(factory=list, kw_only=True)
    '''uvs data, [[(u1, v1), ...], ...]. By default, only will the first uvs data be sent to shader. 
    You can modify this in other mesh classes.'''
    uv_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of uvs after data is cleared by `only_keep_gpu_data`'''
    uv_components_counts: Union[int, List[int]] = attrib(default=2, kw_only=True)
    '''number of uv components. Default is 2. If you have more than 2 uv components, you have to specify this field.'''
    
    tangents: List[Tuple[float, float, float]] = attrib(factory=list, kw_only=True)
    '''tangents data, [(x1, y1, z1), ...]'''
    tangent_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of tangents after data is cleared by `only_keep_gpu_data`'''
    bitangents: List[Tuple[float, float, float]] = attrib(factory=list, kw_only=True)
    '''bitangents data, [(x1, y1, z1), ...]'''
    bitangent_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of bitangents after data is cleared by `only_keep_gpu_data`'''
    
    colors: List[Tuple[float, float, float, float]] = attrib(factory=list, kw_only=True)
    '''colors data, [(r1, g1, b1, a1), ...]'''
    color_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of colors after data is cleared by `only_keep_gpu_data`'''
    
    indices: Union[List[List[int]], List[int]] = attrib(factory=list, kw_only=True)
    '''indices data, [index1, index2, ...]. If not empty, the mesh will be drawn with indices. Otherwise, it will be drawn with vertices.'''
    indices_count: int = attrib(default=0, kw_only=True)
    '''internal use only. For record the number of indices after data is cleared by `only_keep_gpu_data`'''
    
    materials: List[dict] = attrib(factory=list, kw_only=True)
    '''materials data, [material1, material2, ...]. Note that this material data is just dict type, not class `Material`'''
    inner_meshes: List[InnerMesh] = attrib(factory=list, kw_only=True)
    '''inner meshes. Specify this field when there are more than 1 targets inside this mesh data.'''
    
    vao: Optional[int] = attrib(default=None, kw_only=True)
    '''Vertex Array Object. `None` means not sent to GPU. Otherwise, it is the buffer ID.'''
    ebo: Optional[int] = attrib(default=None, kw_only=True)
    '''Element Buffer Object. `None` means not sent to GPU. Otherwise, it is the buffer ID. You may ignore this if you are not using indices.'''
    vbos: List[int] = attrib(factory=list, kw_only=True)
    '''Vertex Buffer Object. `None` means not sent to GPU. Otherwise, it is the buffer ID.'''
    draw_mode: Optional[PrimitiveDrawingType] = PrimitiveDrawingType.TRIANGLES
    '''The OpenGL drawing mode of the mesh. `None` means unknown. It should be set when `load` is called'''
    only_keep_gpu_data: bool = attrib(default=True, kw_only=True)
    '''If True, all vertices data(including normals, ...) will be cleared after sent to GPU. Otherwise, the vertices data will be kept in memory.'''
    generate_vertex_ids: bool = attrib(default=True, kw_only=True)
    '''For Stable-Rendering to do latent tracing(could also be done by texture UV.)
    If indices exists, ID values will not be generated, because the indices can be used as ID values.'''
    cullback: bool = attrib(default=True, kw_only=True)
    '''If True, the mesh will be culled backface. Default is False.'''
    
    # internal use
    meshID: Final[int] = attrib(factory=_get_mesh_id, init=False)
    '''internal integer id for this mesh.'''
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.position_count == 0 and len(self.positions) > 0:
            self.position_count = len(self.positions)
        if self.indices_count == 0 and len(self.indices) > 0:
            if isinstance(self.indices[0], list):
                self.indices_count = len(self.indices) * len(self.indices[0])
            else:
                self.indices_count = len(self.indices)
    
    @property
    def loaded(self):
        '''
        Check if the mesh data is sent to GPU.
        You can override this method in child class.
        '''
        return self.vao is not None
    
    @property
    def has_indices(self):
        return len(self.indices)>0 or self.indices_count>0
    @property
    def has_normals(self):
        return len(self.normals) > 0 or self.normal_count > 0
    @property
    def has_uvs(self):
        return len(self.uvs) > 0 or self.uv_count > 0
    @property
    def has_tangents(self):
        return len(self.tangents) > 0 or self.tangent_count > 0
    @property
    def has_bitangents(self):   
        return len(self.bitangents) > 0 or self.bitangent_count > 0
    @property
    def has_colors(self):
        return len(self.colors) > 0 or self.color_count > 0

    def draw(self, target:Optional[int]=None):
        '''
        Draw the mesh. Override this function to implement drawing the mesh.
        Make sure you have called Material.use() before calling this function. (So that textures & shaders are ready)
        
        Args:
            - target: Only for those objs having multiple inner meshes. If None, draw the whole mesh. Otherwise, draw the specified target.
        '''
        if not self.loaded:
            raise Exception('Data is not sent to GPU')
        if self.draw_mode is None:
            raise Exception('drawMode is not set')
        
        gl.glBindVertexArray(self.vao)

        if target is None: # no material
            if len(self.inner_meshes) == 0: 
                if self.has_indices:
                    gl.glDrawElements(self.draw_mode.value, self.indices_count, gl.GL_UNSIGNED_INT, None)
                else:
                    gl.glDrawArrays(self.draw_mode.value, 0, self.position_count)
            else:
                for i in range(len(self.inner_meshes)):
                    self.draw(i)
        else:
            if target >= len(self.inner_meshes):
                if target == 0 and len(self.inner_meshes) == 0:
                    return self.draw(None) # no inner mesh, draw the whole mesh
                if is_dev_mode():
                    raise Exception(f'incorrect material slot: {target}')
                return # incorrect material slot. skip this mesh in release mode
            inner = self.inner_meshes[target]
            if self.has_indices:
                gl.glDrawElements(self.draw_mode.value, inner.indices_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(inner.from_index * 4))
            else:
                gl.glDrawArrays(self.draw_mode.value, inner.from_vertex, inner.vertex_count)

        gl.glBindVertexArray(0)
        
    def load(self):
        '''send mesh to OpenGL'''
        if self.loaded:
            return
        
        super().load()  # just for printing debug info

        if len(self.positions) == 0:
            raise Exception(f'Mesh `{self.name}` has no data to send to GPU')
        
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        
        if self.has_indices:
            indices_data = np.array(self.indices, dtype=np.uint32).flatten()
            if self.indices_count == 0:
                self.indices_count = indices_data.shape[0]   # prevent from being cleared which makes indices_count to be 0

            self.ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_data.nbytes, indices_data.data, gl.GL_STATIC_DRAW)
        
        # position (0)
        if self.position_count == 0 and not self.has_indices:
            self.position_count = len(self.positions)
        vertices_data = np.array(self.positions, dtype=np.float32).flatten()
        vertex_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data.data, gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
        self.vbos.append(vertex_vbo)
        
        # normal (1)
        self.normal_count = len(self.normals)
        if self.has_normals:
            normals_data = np.array(self.normals, dtype=np.float32).flatten()
            normal_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, normal_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, normals_data.nbytes, normals_data.data, gl.GL_STATIC_DRAW)
            
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
            self.vbos.append(normal_vbo)
            
        # tangent (2)
        self.tangent_count = len(self.tangents)
        if self.has_tangents:
            tangent_data = np.array(self.tangents, dtype=np.float32).flatten()
            tangent_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, tangent_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, tangent_data.nbytes, tangent_data.data, gl.GL_STATIC_DRAW)
            
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
            self.vbos.append(tangent_vbo)
        
        # bitangent (3)
        self.bitangent_count = len(self.bitangents)
        if self.has_bitangents:
            bitangent_data = np.array(self.bitangents, dtype=np.float32).flatten()
            bitangent_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bitangent_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, bitangent_data.nbytes, bitangent_data.data, gl.GL_STATIC_DRAW)
            
            gl.glEnableVertexAttribArray(3)
            gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
            self.vbos.append(bitangent_vbo)
        
        # color (4)
        if self.has_colors:
            self.color_count = len(self.colors)
            color_data = np.array(self.colors, dtype=np.float32).flatten()
            color_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, color_data.nbytes, color_data.data, gl.GL_STATIC_DRAW)
            
            gl.glEnableVertexAttribArray(4)
            channel_count = len(self.colors[0])
            gl.glVertexAttribPointer(4, channel_count, gl.GL_FLOAT, gl.GL_FALSE, channel_count * 4, None)
            self.vbos.append(color_vbo)
                    
        # id (5)
        if self.generate_vertex_ids:
            if self.has_indices:
                vertex_ids = indices_data
            else:
                vertex_ids = np.array([i for i in range(self.position_count)], dtype=np.int32).flatten()
            
            id_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, id_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_ids.nbytes, vertex_ids.data, gl.GL_STATIC_DRAW)
            
            gl.glEnableVertexAttribArray(5)
            gl.glVertexAttribPointer(5, 1, gl.GL_INT, gl.GL_FALSE, 1 * 4, None)
            self.vbos.append(id_vbo)
        
        # uv (start from 8)
        point_offset = 8
        self.uv_count = len(self.uvs)
        if self.has_uvs:
            for i in range(len(self.uvs)): 
                uv_data = np.array(self.uvs[i], dtype=np.float32).flatten()
                component_count = self.uv_components_counts if isinstance(self.uv_components_counts, int) else self.uv_components_counts[i]
                uv_vbo = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, uv_vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, uv_data.nbytes, uv_data.data, gl.GL_STATIC_DRAW)
                
                gl.glEnableVertexAttribArray(point_offset + i)
                gl.glVertexAttribPointer(point_offset + i, component_count, gl.GL_FLOAT, gl.GL_FALSE, component_count * 4, None)
                self.vbos.append(uv_vbo)
            
        if self.only_keep_gpu_data:
            self.positions.clear()
            self.normals.clear()
            self.uvs.clear()
            self.tangents.clear()
            self.bitangents.clear()
            self.colors.clear()
            self.indices.clear()
        
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
    
    @classmethod
    def Load(cls,
             path: Union[str, Path],
             name: Optional[str]=None,
             only_keep_gpu_data:bool=False,
             generate_vertex_ids:bool=False,
             **kwargs,
             ):
        '''
        load a mesh from file by default method(assimp_py).

        Args:
            * path: path of the mesh file
            * name: name is used to identify the mesh. If not specified, the name will be the path of the mesh file
            * only_keep_gpu_data: if True, the vertices data will be cleared after sent to GPU. Otherwise, the vertices data will be kept in memory.
            * generate_vertex_ids: if True, the mesh will generate vertex id for each vertex. This is for stable-renderer to do latent tracing.
        '''
        process_flags = (assimp_py.Process_Triangulate | assimp_py.Process_CalcTangentSpace | assimp_py.Process_JoinIdenticalVertices)
        scene = assimp_py.ImportFile(str(path), process_flags)
        
        all_positions = [] # list[tuple[x,y,z],...] (positions)
        all_normals = []
        all_uvs = []    # list[list[tuple[u,v],..],...]
        all_tangents = []
        all_bitangents = []
        all_colors = []
        all_materials = []  # list[dict, ...]
        all_indices = []    # list[tuple[index1, index2, ...], ...]
        inner_meshes = []   # list[InnerMesh, ...]
        uv_components_counts: Union[int, List[int]] = []
        for mesh in scene.meshes:
            if not uv_components_counts:
                counts = [num_component for num_component in mesh.num_uv_components]    # currently only support same num of texcoords among all meshes
                if len(counts) == 0:
                    uv_components_counts = 2
                else:
                    uv_components_counts = counts
                
            vertices = mesh.vertices    # positions must be present
            normals = getattr(mesh, 'normals', [])
            texcoords = getattr(mesh, 'texcoords', [])
            tangents = getattr(mesh, 'tangents', [])
            bitangent = getattr(mesh, 'bitangents', [])
            color = getattr(mesh, 'colors', [])
            indices = getattr(mesh, 'indices', [])
            mat = scene.materials[mesh.material_index] if len(scene.materials) > 0 else None
            
            inner_mesh = InnerMesh(material_name=mat['NAME'] if mat else None,
                                   from_vertex=len(all_positions), 
                                   vertex_count=len(vertices),
                                   from_index=len(all_indices) * 3,
                                   indices_count=len(indices) * 3,
                                   material_index=mesh.material_index)
            inner_meshes.append(inner_mesh)
            
            all_positions.extend(vertices)
            all_normals.extend(normals)
            all_tangents.extend(tangents)
            all_bitangents.extend(bitangent)
            all_colors.extend(color)
            all_indices.extend([[ind + inner_mesh.from_vertex for ind in face] for face in indices])
            all_materials.append(mat) if mat else None
            
            for j, texcoord in enumerate(texcoords):
                uv_components_count = mesh.num_uv_components[j] if isinstance(mesh.num_uv_components, list) else mesh.num_uv_components
                if len(all_uvs) <= j:
                    all_uvs.append([])
                all_uvs[j].extend([tuple(uv[:uv_components_count]) for uv in texcoord])
        
        data = {
            'name': name,
            'positions': all_positions,
            'normals': all_normals,
            'uvs': all_uvs,
            'tangents': all_tangents,
            'bitangents': all_bitangents,
            'colors': all_colors,
            'indices':all_indices,
            'position_count': len(all_positions),
            'indices_count': len(all_indices) * 3,
            'materials': all_materials,
            'inner_meshes': inner_meshes,
            'uv_components_counts': uv_components_counts,
            'only_keep_gpu_data': only_keep_gpu_data,
            'generate_vertex_ids': generate_vertex_ids,
        }
        data.update(kwargs)
        mesh_obj = cls(**data)
        return mesh_obj

    def clear(self):
        super().clear() # just for printing debug info
        for vbo in self.vbos:
            buffer = np.array([vbo], dtype=np.uint32)
            gl.glDeleteBuffers(1, buffer)
        self.vbos.clear()
        
        if self.ebo is not None:
            buffer = np.array([self.ebo], dtype=np.uint32)
            gl.glDeleteBuffers(1, buffer)
            self.ebo = None
        
        if self.vao is not None:
            buffer = np.array([self.vao], dtype=np.uint32)
            gl.glDeleteVertexArrays(1, buffer)
            self.vao = None
        
        self.positions.clear()
        self.normals.clear()
        self.uvs.clear()
        self.tangents.clear()
        self.bitangents.clear()
        self.colors.clear()
        self.indices.clear()
        self.materials.clear()
        self.inner_meshes.clear()
        
        self.position_count = 0
        self.indices_count = 0
        self.normal_count = 0
        self.color_count = 0
        self.tangent_count = 0
        self.bitangent_count = 0
        self.uv_count = 0
        self.uv_components_counts = 0
        
    # region default meshes
    @staticmethod
    def Plane(edge:int = 1)->'Mesh':
        '''get a plane mesh. Vertex per edge = edge+1'''
        name = f'__PLANE_MESH_EDGE_{edge}__'
        if plane_mesh := Mesh.Find(name=name):
            return plane_mesh
        else:
            return _PlaneMesh(name=name, edge=edge)

    @staticmethod
    def Sphere(segment=32)->'Mesh':
        '''
        get a sphere mesh with specified segment
        :param segment: the segment of the sphere. More segments means more smooth sphere.
        :return: Mesh object
        '''
        name = f'__SPHERE_MESH_SEGMENT_{segment}__'
        if sphere_mesh := Mesh.Find(name=name):
            return sphere_mesh
        else:
            return _SphereMesh(name=name, segment=segment)
    # endregion

# region basic shapes
@attrs(eq=False, repr=False)
class _PlaneMesh(Mesh):

    edge: int = attrib(default=1)
    '''edge of the plane. Vertex per edge = edge+1'''
    generate_vertex_ids: bool = attrib(default=True)
    '''plane is default to generate vertex ids for latent tracing'''
    
    def __attrs_post_init__(self):
        xcoords = np.linspace(-0.5, 0.5, self.edge+1)
        zcoords = np.linspace(-0.5, 0.5, self.edge+1)
        positions = []
        normals = []
        uvs = []
        tangents = []
        bitangents = []
        for z in zcoords:
            for x in xcoords:
                positions.append((x, 0, z))
                normals.append((0, 1, 0))
                uvs.append((x+0.5, z+0.5))
                tangents.append((1, 0, 0))
                bitangents.append((0, 0, 1))
        
        self.positions = positions
        self.normals = normals
        self.uvs = [uvs]
        self.tangents = tangents
        self.bitangents = bitangents
        self.position_count = len(positions)
        self.indices = []
        
        for i in range(self.edge):
            for j in range(self.edge):
                self.indices.append(i*(self.edge+1)+j)
                self.indices.append((i+1)*(self.edge+1)+j)
                self.indices.append((i+1)*(self.edge+1)+j+1)
                
                self.indices.append(i*(self.edge+1)+j)
                self.indices.append((i+1)*(self.edge+1)+j+1)
                self.indices.append(i*(self.edge+1)+j+1)

        self.draw_mode = PrimitiveDrawingType.TRIANGLES
        
        super().__attrs_post_init__()

@attrs(eq=False, repr=False)
class _SphereMesh(Mesh):

    segment: int = attrib(default=32)
    '''segment of the sphere. More segments means more smooth sphere.'''
    generate_vertex_ids: bool = attrib(default=True)
    '''plane is default to generate vertex ids for latent tracing'''
    
    def __attrs_post_init__(self):
        segment = self.segment
        positions = []
        normals = []
        uvs = []
        tangents = []
        bitangents = []
        
        for i in range(segment+1):
            for j in range(segment+1):
                xSegment = j / segment
                ySegment = i / segment
                xPos = math.cos(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                yPos = math.cos(ySegment * math.pi)
                zPos = math.sin(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                positions.append((xPos, yPos, zPos))
                normals.append((xPos, yPos, zPos))
                uvs.append((xSegment, ySegment))
                
                tx = -math.sin(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                ty = 0
                tz = math.cos(xSegment * 2 * math.pi) * math.sin(ySegment * math.pi)
                tangents.append((tx, ty, tz))

                bx = math.cos(xSegment * 2 * math.pi) * math.cos(ySegment * math.pi)
                by = -math.sin(ySegment * math.pi)
                bz = math.sin(xSegment * 2 * math.pi) * math.cos(ySegment * math.pi)
                bitangents.append((bx, by, bz))
                
        indices = []
        for i in range(segment):
            for j in range(segment):
                indices.append((i + 1) * (segment + 1) + j)
                indices.append(i * (segment + 1) + j)
        
        self.positions = positions
        self.normals = normals
        self.uvs = [uvs]
        self.tangents = tangents
        self.bitangents = bitangents
        self.position_count = len(positions)
        self.indices = indices
        self.draw_mode = PrimitiveDrawingType.TRIANGLE_STRIP
        
        super().__attrs_post_init__()

# TODO: Cube Mesh & other common shapes
# endregion

__all__ = ['Mesh']