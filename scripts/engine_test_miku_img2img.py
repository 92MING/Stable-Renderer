import os, sys
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(folder_path, 'source'))

from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController
from engine.static import Mesh, Material_MTL
from common_utils.path_utils import *

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(2.5 * self.engine.RuntimeManager.DeltaTime)
    
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
           
            
    img2img_workflow_path = EXAMPLE_WORKFLOWS_DIR / 'miku-img2img-example.json'
    Sample.Run(winSize=(512, 512),
               mapSavingInternal=1,
               needOutputMaps=False,
               saveSDColorOutput=False,
               disableComfyUI=False,
               workflow=img2img_workflow_path)
