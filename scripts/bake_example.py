import os, sys
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(folder_path, 'source'))

from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController, CorrMapRenderer, SpriteInfo, EqualIntervalRotation
from engine.static import Mesh, Material_MTL, CorrespondMap, EngineMode
from common_utils.path_utils import *


if __name__ == '__main__':

    class Sample(Engine):
        def beforePrepare(self):
            mikuPath = os.path.join(EXAMPLE_3D_MODEL_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'), alias='miku', cullback=False) 
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials
            
            camera = GameObject('Camera', position=[0, 0.68, 2.3])
            camera.addComponent(Camera, bgPrompt='no background')
            camera.transform.lookAt([0, 0.68, 0])
            #camera.addComponent(CameraController, defaultPos=[0, 0.68, 2.3], defaultLookAt=[0, 0.68, 0])
            
            miku = GameObject('miku', position=[0, 0.02, 0], scale=0.065)
            meshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(EqualIntervalRotation, interval=18)
            
            self.corrmap = CorrespondMap()
            corrmap_obj = GameObject('corrmap', position=[0, 0.68, 0], scale=0.85)
            corrmap_obj.addComponent(SpriteInfo, auto_spriteID=True, prompt='miku, 1 girl, long blue hair, waifu, white dresses')
            corrmap_obj.addComponent(CorrMapRenderer, corrmaps=self.corrmap)
            corrmap_obj.addComponent(EqualIntervalRotation, interval=18)
            
        def beforeRelease(self):
            self.corrmap.dump(OUTPUT_DIR, name='miku_corrmap')
    
    #baking_workflow = EXAMPLE_WORKFLOWS_DIR / 'bake.json'
    #baking_workflow = EXAMPLE_WORKFLOWS_DIR / 'no-normal-bake.json'
    #baking_workflow = EXAMPLE_WORKFLOWS_DIR / 'no-control-bake.json'
    baking_workflow = EXAMPLE_WORKFLOWS_DIR / 'no-mask-prompt-bake.json'
    Sample.Run(winSize=(512, 512),
               mode = EngineMode.BAKE,
               mapSavingInterval=1,
               baking_interval= 8,
               needOutputMaps=False,
               disableComfyUI=False,
               diffuse_workflow=baking_workflow)
