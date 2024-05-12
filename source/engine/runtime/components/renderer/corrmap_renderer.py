'''special renderer for AI-Baking'''
import glm
from functools import partial
from typing import Iterable, Union, TYPE_CHECKING, Optional

from engine.static.mesh import Mesh
from engine.static.material import Material, DefaultTextureType
from engine.static.enums import RenderOrder, EngineMode, RenderMode
from .renderer import Renderer
from ..ai import SpriteInfo
from common_utils.global_utils import GetOrAddGlobalValue, SetGlobalValue
from common_utils.debug_utils import EngineLogger

if TYPE_CHECKING:
    from ...gameObj import GameObject
    from common_utils.stable_render_utils import CorrespondMap


_DEFAULT_CORRMAP_MATERIAL: Material = GetOrAddGlobalValue('__DEFAULT_CORRMAP_MATERIAL__', None)   # type: ignore
def _get_default_corrmap_material()->Material:
    global _DEFAULT_CORRMAP_MATERIAL
    if _DEFAULT_CORRMAP_MATERIAL is None:
        _DEFAULT_CORRMAP_MATERIAL = Material.DefaultTransparentMaterial()
        SetGlobalValue('__DEFAULT_CORRMAP_MATERIAL__', _DEFAULT_CORRMAP_MATERIAL)
    return _DEFAULT_CORRMAP_MATERIAL 

_DEFAULT_CORRMAP_MESH: Mesh = GetOrAddGlobalValue('__DEFAULT_CORRMAP_MESH__', None)   # type: ignore
def _get_default_corrmap_mesh():
    global _DEFAULT_CORRMAP_MESH
    if _DEFAULT_CORRMAP_MESH is None:
        _DEFAULT_CORRMAP_MESH = Mesh.Sphere()
        SetGlobalValue('__DEFAULT_CORRMAP_MESH__', _DEFAULT_CORRMAP_MESH)
    return _DEFAULT_CORRMAP_MESH

def _DEFAULT_USE_TEXCOORD_ID(m: Mesh):
    if m == _get_default_corrmap_mesh():
        return True # the default sphere mesh will use texcoord as ID by default
    return False

class CorrMapRenderer(Renderer):
    '''
    Renderer for Correspondence Map.
    This could be both for rendering & baking purpose.
    '''
    
    RequireComponent = (SpriteInfo, )
    
    def __init__(self, 
                 gameObj: "GameObject", 
                 enable=True,
                 corrmaps: Union["CorrespondMap", Iterable["CorrespondMap"], None] = None,
                 mesh: Optional["Mesh"] = None,
                 materials:Union["Material", Iterable["Material"], None] = None,
                 use_texcoord_id: bool = _DEFAULT_USE_TEXCOORD_ID   # type: ignore
                 ):
        '''
        Note: `SpriteInfo` is required for this component's gameobject.
        
        Args:
            - gameObj: the GameObject that this component belongs to.
            - enable: if this component is enabled.
            - corrmaps: the target Correspondence Maps. Note that the number of corrmap should be equal to the number of materials.
            - mesh: the mesh used for mapping the vertex to pixel. Default to be a sphere.
            - materials: A sequence of materials. If it is provided, material ID will be used for separating the materials.
                         Default to be a material with transparent order(since there should be transparent part in corrmap).
            - use_tex_coord_id: if True, even the mesh has vertexID, UV texcoord will be used for pixel tracing instead.
                                Default to be true for sphere mesh mapper.
            - spriteID: force the spriteID in shader to be a specific value. It can be None if it is offered in the `corrmap`.
        '''
        super().__init__(gameObj=gameObj, enable=enable, materials=materials)
        
        from common_utils.stable_render_utils import CorrespondMap
        
        if corrmaps is None:
            self.corrmaps = []
        elif isinstance(corrmaps, CorrespondMap):
            self.corrmaps = [corrmaps,]
        elif isinstance(corrmaps, (list, tuple)):
            self.corrmaps = list(corrmaps)
        else:   # none
            raise ValueError('corrmap should be a CorrespondMap object or a list of CorrespondMap objects.')
        self.mesh = mesh or _get_default_corrmap_mesh()
        self.use_tex_coord_id = use_texcoord_id
        if not self.materials:
            self._materials = [_get_default_corrmap_material(), ]
        if self.use_tex_coord_id == _DEFAULT_USE_TEXCOORD_ID:
            self.use_tex_coord_id = _DEFAULT_USE_TEXCOORD_ID(self.mesh)
  
    @property
    def sprite(self):
        '''the sprite object contained in the `SpriteInfo` component.'''
        if not (sprite_info:=self.gameObj.getComponent(SpriteInfo)):
            return None
        return sprite_info.sprite
    
    def start(self):
        if self.engine.Mode == EngineMode.BAKE:
            return
        for i, mat in enumerate(self.materials):
            if i>=len(self.corrmaps):
                EngineLogger.warn('The number of corrmaps is less than the number of materials. Please make sure they are matched.')
                break
            if not mat.hasDefaultTexture(DefaultTextureType.CorrespondMap):
                mat.addDefaultTexture(self.corrmaps[i], DefaultTextureType.CorrespondMap)
        
    def _drawAvailable(self):
        return (self.sprite is not None and \
                len(self.corrmaps) == len(self.materials) and \
                self.mesh is not None)
    
    def _renderTask(self, 
                    modelM: glm.mat4, 
                    mesh: "Mesh", 
                    corrmap: "CorrespondMap",
                    material: "Material", 
                    use_tex_coord_id=True,
                    slot: Optional[int]=None):
        '''For submitting to RenderManager'''
        if self.sprite is None:
            return
        
        self.engine.RuntimeManager.UpdateUBO_ModelMatrix(modelM)
        spriteID = self.sprite.spriteID
        
        material.use()  # this will send corrmap's as array texture to shader in baked mode
        material.shader.setUniform('spriteID', spriteID)
        material.shader.setUniform('corrmap_k', corrmap.k)
        material.shader.setUniform('useTexcoordAsID', int(use_tex_coord_id and mesh.has_uvs))
        
        if self.engine.Mode == EngineMode.BAKE:
            material.shader.setUniform('renderMode', int(RenderMode.BAKING.value))
        else:
            material.shader.setUniform('renderMode', int(RenderMode.BAKED.value))
        
        mesh.draw(target=slot)

    def _check_and_get_order_factor(self):
        from ..camera import Camera
        main_cam = Camera.MainCamera()
        
        if main_cam is not None:
            self_pos_under_cam_system = main_cam.transform.inverseTransformPoint(self.transform.position)
            return (self_pos_under_cam_system.z > 0), self_pos_under_cam_system.z + 1
        
        return False, 1

    def _draw(self):
        '''Drawing task will be deferred to RenderManager'''
        
        should_render, main_cam_z = self._check_and_get_order_factor()
        if not should_render:
            return
        transform_matrix = self.transform.globalTransformMatrix
        mesh = self.mesh
        use_tex_coord_id = self.use_tex_coord_id
        
        if not mesh or not self.corrmaps:
            return
        
        for i, mat in enumerate(self.materials):
            corrmap = self.corrmaps[i]
            if RenderOrder.OPAQUE <= mat.render_order < RenderOrder.TRANSPARENT:    # opaque
                if not should_render:
                    continue
                order = mat.render_order - 1 / main_cam_z   # opaque do from near to far
            elif mat.render_order < RenderOrder.OVERLAY:    # transparent
                if not should_render:
                    continue
                order = mat.render_order + 1 / main_cam_z   # transparent do from far to near
            
            self.engine.RenderManager.AddGBufferTask(order=order,
                                                     mesh=self.mesh,
                                                     shader=mat.shader,
                                                     task=partial(self._renderTask, transform_matrix, mesh, corrmap, mat, use_tex_coord_id, slot=i))
            if self.sprite is not None:
                self.engine.RenderManager.SubmitCorrmap(self.sprite.spriteID, mat.materialID, corrmap)

        

__all__ = ['CorrMapRenderer']