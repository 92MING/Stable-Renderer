import glm

from functools import partial
from typing import Iterable, Union, Sequence, Optional, TYPE_CHECKING, Callable, Any
from .renderer import Renderer
from engine.static.enums import RenderOrder, RenderMode
from common_utils.stable_render_utils import get_new_spriteID
from ..ai import SpriteInfo

if TYPE_CHECKING:
    from engine.static.material import Material, Material_MTL
    from engine.static.mesh import Mesh


class MeshRenderer(Renderer):
    '''Renderer for mesh object. It will draw the mesh with the materials.'''

    def __init__(self, 
                 gameObj, 
                 enable=True,
                 materials: Union["Material", Iterable["Material"], None] = None,
                 mesh: Optional["Mesh"] = None,
                 use_texcoord_id: bool = False,
                 spriteID: Optional[int] = None,
                 defer_render_task: Optional[Callable[..., Any]] = None,
                 ):
        '''
        Args:
            - gameObj: the GameObject that this component belongs to.
            - enable: if this component is enabled.
            - materials: a sequence of materials.
            - mesh: the mesh object to render.
            - use_tex_coord_id: if True, even the mesh has vertexID, UV texcoord will be used for pixel tracing instead.
                                Default to be true for sphere mesh mapper.
            - spriteID: force the spriteID in shader to be a specific value. It can be None if it is offered in the `corrmap`.
            - defer_render_task: a callable object that will be called when rendering. It should have the same signature as `_renderTask`.
        '''
        super().__init__(gameObj=gameObj, enable=enable, materials=materials)
        self.mesh = mesh
        self.use_texcoord_id = use_texcoord_id
        self.force_spriteID = spriteID if spriteID is not None else get_new_spriteID()
        self.defer_render_task = defer_render_task
    
    @property
    def spriteID(self):
        '''the sprite object contained in the `SpriteInfo` component.'''
        if self.force_spriteID is not None:
            return self.force_spriteID
        if not (sprite_info:=self.gameObj.getComponent(SpriteInfo)):
            return None
        return sprite_info.sprite.spriteID

    def load_MTL_Materials(self, mats: Union["Material_MTL", Sequence["Material_MTL"]]):
        '''
        Load MTL materials and add as materials.
        This method is for loading the case that .obj file contains MTL materials.

        Args:
            mats: a sequence of materials. The key is the MTL name, and the value is the Material_MTL object.
        '''
        from engine.static.material import Material, Material_MTL
        if self.mesh is None:
            raise ValueError('No mesh loaded. Cannot load MTL materials.')

        if isinstance(mats, Material_MTL):
            mats = [mats]
        matDict = {m.real_name: m for m in mats}
        for origin_mtl_data in self.mesh.materials:
            name = origin_mtl_data.get('NAME', origin_mtl_data.get('name', origin_mtl_data.get('Name', None)))
            if name is not None:
                if name in matDict:
                    self.addMaterial(matDict[name], duplicateCheck=False)
                else:
                    self.addMaterial(Material.DefaultDebugMaterial(), duplicateCheck=False) # parts without material will be pink

    def _renderTask(self, modelM: glm.mat4, material: "Material", mesh: Optional["Mesh"], slot: Optional[int]=None):
        '''For submitting to RenderManager'''
        self.engine.RuntimeManager.UpdateUBO_ModelMatrix(modelM)
        
        material.use()
        material.shader.setUniform('renderMode', int(RenderMode.NORMAL.value))  # for AI rendering, plz use `CorrMapRenderer`
        material.shader.setUniform('spriteID', self.spriteID or 0)  # 0 means no sprite
        material.shader.setUniform('useTexcoordAsID', int(self.use_texcoord_id and self.mesh is not None and self.mesh.has_uvs))
        
        if mesh is not None:
            material.shader.setUniform("hasVertexColor", int(mesh.has_colors))  # vertex color will be used only when texture is not given
            mesh.draw(slot)
        material.unbind()
            
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
        transform_matrix = self.transform.globalTransformMatrix
        mesh = self.mesh
        
        for i, mat in enumerate(self.materials):    # do nothing is no materials
            if RenderOrder.OPAQUE <= mat.render_order < RenderOrder.TRANSPARENT:    # opaque
                if not should_render:
                    continue
                order = mat.render_order - 1 / main_cam_z   # opaque do from near to far
            elif mat.render_order < RenderOrder.OVERLAY:    # transparent
                if not should_render:
                    continue
                order = mat.render_order + 1 / main_cam_z   # transparent do from far to near
            else:   # overlay
                order = mat.render_order
            
            self.engine.RenderManager.AddGBufferTask(order=order,
                                                     mesh=mesh,
                                                     shader=mat.shader,
                                                     task=partial(self._renderTask, transform_matrix, mat, mesh, slot=i)
                                                     )
        
        if self.defer_render_task is not None:
            self.engine.RenderManager.AddDeferRenderTask(self.defer_render_task)

    def _drawAvailable(self):
        return super()._drawAvailable() and self.mesh is not None


__all__ = ['MeshRenderer']