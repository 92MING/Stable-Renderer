import os, sys
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(folder_path, 'source'))

from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController
from engine.static import Mesh, Material, Material_MTL, Texture, DefaultTextureType, TextureWrap
from common_utils.path_utils import *

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.5)
    
    class Sample(Engine):
        def beforePrepare(self):
            mikuPath = os.path.join(EXAMPLE_3D_MODEL_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'), alias='miku') 
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials
            
            camera = GameObject('Camera')
            camera.addComponent(Camera)
            camera.addComponent(CameraController, defaultPos=[1.3, 2.8, 1.3], defaultLookAt=[0, 2.8, 0])

            miku = GameObject('miku', position=[0, 0, 0], scale=[0.16, 0.16, 0.16])
            meshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(AutoRotation)
            
            debug_mat = Material.DefaultOpaqueMaterial()
            debug_mat.addDefaultTexture(
                Texture.Load(
                    os.path.join(EXAMPLE_3D_MODEL_DIR, 'debug', 'debug_uv_texture.jpg'),
                    s_wrap=TextureWrap.CLAMP_TO_EDGE,
                ),
                DefaultTextureType.DiffuseTex
            )
            
            ball = GameObject('ball', position=[2, 0.5, 2], scale=[0.5, 0.5, 0.5])
            ball_meshRenderer = ball.addComponent(MeshRenderer, mesh=Mesh.Sphere())
            ball_meshRenderer.addMaterial(debug_mat)
            
            plane = GameObject('plane', position=[0, 0, 0], scale=[5,5,5])
            plane_meshRenderer:MeshRenderer = plane.addComponent(MeshRenderer, mesh=Mesh.Plane())
            plane_meshRenderer.addMaterial(debug_mat)

    Sample.Run(debug=False,
               winSize=(512, 512),
               mapSavingInterval=4,
               disableComfyUI=True,
               needOutputMaps=False,)
