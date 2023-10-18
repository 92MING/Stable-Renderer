from .renderer import Renderer
from static.material import Material
from static.mesh import Mesh
from functools import partial

class MeshRenderer(Renderer):
    def __init__(self, gameObj, enable=True, material: Material = None, mesh: Mesh = None, ):
        super().__init__(gameObj, enable, material)
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def _renderTask(self, modelM, material, mesh):
        self.engine.RenderManager.UpdateUBO_ModelMatrix(modelM)
        material.use()
        mesh.draw()
    def _draw(self):
        self.engine.RenderManager.AddRenderTask(order=self.material.renderOrder,
                                                mesh=self.mesh,
                                                shader=self.material.shader,
                                                task=partial(self._renderTask, self.transform.globalTransformMatrix, self.material, self.mesh))
    def _drawAvailable(self):
        return super()._drawAvailable() and self._mesh is not None

__all__ = ['MeshRenderer']