import os, sys
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(folder_path, 'source'))

from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.runtime.components import Camera, MeshRenderer, CameraController, SpriteInfo
from engine.engine import Engine
from engine.static import Mesh, Material_MTL
from common_utils.path_utils import *
from common_utils.stable_render_utils import EnvPrompt

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(2.5 * self.engine.RuntimeManager.DeltaTime)
    
    class Sample(Engine):
        def beforePrepare(self):
            mikuPath = os.path.join(EXAMPLE_3D_MODEL_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'), alias='miku') 
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials
            
            env_prompt = EnvPrompt('no background', negative_prompt="watermark")
            camera = GameObject('Camera')
            camera.addComponent(Camera, bgPrompt=env_prompt)
            camera.addComponent(CameraController, defaultPos=[1.3, 2.8, 1.3], defaultLookAt=[0, 2.8, 0])

            miku = GameObject('miku', position=[0, 0, 0], scale=[0.16, 0.16, 0.16])
            meshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(AutoRotation)
            miku.addComponent(SpriteInfo, auto_spriteID=True, prompt='miku')
    
    # workflow_path = EXAMPLE_WORKFLOWS_DIR / 'miku-img2img-example.json'
    # workflow_path = EXAMPLE_WORKFLOWS_DIR / 'miku-direct-latent.json'
    workflow_path = EXAMPLE_WORKFLOWS_DIR / 'new-miku-control.json'
    Sample.Run(winSize=(512, 512),
               mapSavingInternal=1,
               needOutputMaps=False,
               saveSDColorOutput=False,
               disableComfyUI=False,
               verbose=True,
               diffuse_workflow=workflow_path)
