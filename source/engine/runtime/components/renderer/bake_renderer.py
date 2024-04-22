'''special renderer for AI-Baking'''

from functools import partial
from typing import Iterable, Union, TYPE_CHECKING, Optional
from .mesh_renderer import MeshRenderer
from engine.static.material import Material
from engine.static.mesh import Mesh
if TYPE_CHECKING:
    from engine.static.mesh import Mesh

class BakeRenderer(MeshRenderer):
    '''Renderer for AI-Baking.'''

    baking_k: int
    '''
    The value that determines how many pixels per edge should be baked during AI-Baking.
    Larger value results in more smooth result, but consumes more VRAM, i.e. baked texture number = k^2 
    '''
    
    def __init__(self, 
                 gameObj, 
                 enable=True,
                 materials: Union[Material, Iterable[Material]] = None,
                 mesh: Mesh = None, 
                 baking_k: int = 3,
                 ):
        super().__init__(gameObj=gameObj, enable=enable, materials=materials, mesh=mesh)
        self.baking_k = baking_k
        
    def _renderTask(self, modelM, material: Material, mesh: "Mesh", slot: Optional[int]=None):
        '''For submitting to RenderManager'''
        self.engine.RuntimeManager.UpdateUBO_ModelMatrix(modelM)
        material.use()
        material.shader.setUniform('isBaking', 1)
        material.shader.setUniform('baking_k', self.baking_k)
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


__all__ = ['BakeRenderer']