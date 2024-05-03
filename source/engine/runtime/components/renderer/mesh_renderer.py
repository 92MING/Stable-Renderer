import glm

from functools import partial
from typing import Iterable, Union, Sequence, Optional, TYPE_CHECKING
from .renderer import Renderer

if TYPE_CHECKING:
    from engine.static.material import Material, Material_MTL
    from engine.static.mesh import Mesh


class MeshRenderer(Renderer):
    '''Renderer for mesh object. It will draw the mesh with the materials.'''

    def __init__(self, 
                 gameObj, 
                 enable=True,
                 materials: Union["Material", Iterable["Material"], None] = None,
                 mesh: Optional["Mesh"] = None
                 ):
        super().__init__(gameObj=gameObj, enable=enable, materials=materials)
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def load_MTL_Materials(self, mats: Union["Material_MTL", Sequence["Material_MTL"]]):
        '''
        Load MTL materials and add as materials.
        This method is for loading the case that .obj file contains MTL materials.

        Args:
            mats: a sequence of materials. The key is the MTL name, and the value is the Material_MTL object.
        '''
        from engine.static.material import Material, Material_MTL
        if self._mesh is None:
            raise ValueError('No mesh loaded. Cannot load MTL materials.')

        if isinstance(mats, Material_MTL):
            mats = [mats]
        matDict = {m.real_name: m for m in mats}
        for origin_mtl_data in self._mesh.materials:
            name = origin_mtl_data.get('NAME', origin_mtl_data.get('name', origin_mtl_data.get('Name', None)))
            if name is not None:
                if name in matDict:
                    self.addMaterial(matDict[name], duplicateCheck=False)
                else:
                    self.addMaterial(Material.DefaultDebugMaterial(), duplicateCheck=False) # parts without material will be pink

    def _renderTask(self, modelM: glm.mat4, material: "Material", mesh: "Mesh", slot: Optional[int]=None):
        '''For submitting to RenderManager'''
        self.engine.RuntimeManager.UpdateUBO_ModelMatrix(modelM)
        material.use()
        mesh.draw(slot)

    def _draw(self):
        '''Drawing task will be deferred to RenderManager'''
        for i, mat in enumerate(self.materials):
            self.engine.RenderManager.AddRenderTask(order=mat.render_order,
                                                    mesh=self.mesh,
                                                    shader=mat.shader,
                                                    task=partial(
                                                        self._renderTask, 
                                                        self.transform.globalTransformMatrix, 
                                                        mat, 
                                                        self.mesh, 
                                                        slot=i
                                                        )
                                                    )

    def _drawAvailable(self):
        return super()._drawAvailable() and self._mesh is not None


__all__ = ['MeshRenderer']