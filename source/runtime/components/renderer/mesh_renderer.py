from .renderer import Renderer
from static.material import Material, Material_MTL
from static.mesh import Mesh, Mesh_OBJ
from functools import partial
from typing import Iterable, Union, Dict

class MeshRenderer(Renderer):
    def __init__(self, gameObj, enable=True,
                 materials: Union[Material, Iterable[Material]] = None,
                 mesh: Mesh = None, ):
        super().__init__(gameObj, enable, materials)
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def load_MTL_Materials(self, matDict: Dict[str, 'Material_MTL']):
        '''Load MTL materials. This method is for loading the case that .obj file contains MTL materials.'''
        if self._mesh is None:
            raise ValueError('No mesh loaded. Cannot load MTL materials.')
        if not isinstance(self._mesh, Mesh_OBJ):
            raise ValueError('Only .obj file can load MTL materials. Current mesh is in format of {}'.format(self._mesh.Format()))
        for mtl_mat_data in self._mesh.materials:
            if mtl_mat_data.materialName in matDict:
                self.addMaterial(matDict[mtl_mat_data.materialName], duplicateCheck=False)
            else:
                self.addMaterial(Material.Debug_Material(), duplicateCheck=False) # the part without material will be pink

    def _renderTask(self, modelM, material, mesh, slot=None):
        '''For submitting to RenderManager'''
        self.engine.RuntimeManager.UpdateUBO_ModelMatrix(modelM)
        material.use()
        mesh.draw(group=slot)
    def _draw(self):
        '''Drawing task will be deferred to RenderManager'''
        for i, mat in enumerate(self.materials):
            self.engine.RenderManager.AddRenderTask(order=mat.renderOrder,
                                                    mesh=self.mesh,
                                                    shader=mat.shader,
                                                    task=partial(self._renderTask, self.transform.globalTransformMatrix, mat, self.mesh, slot=i))
    def _drawAvailable(self):
        return super()._drawAvailable() and self._mesh is not None


__all__ = ['MeshRenderer']