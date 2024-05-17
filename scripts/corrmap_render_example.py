import os, sys
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(folder_path, 'source'))

from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController, CorrMapRenderer, SpriteInfo, EqualIntervalRotation, AutoRotation
from engine.static import Mesh, Material_MTL, CorrespondMap, EngineMode, Texture, Material
from engine.static.enums import TextureFormat, TextureDataType, TextureFilter, TextureWrap, TextureInternalFormat, DefaultTextureType
from common_utils.path_utils import *


if __name__ == '__main__':

    class Sample(Engine):
        
        def beforePrepare(self):
            camera = GameObject('Camera', position=[0, 0.68, 2.3])
            camera.addComponent(Camera, bgPrompt='no background')
            camera.transform.lookAt([0, 0.68, 0])
            camera.addComponent(CameraController, defaultPos=[0, 0.68, 2.3], defaultLookAt=[0, 0.68, 0])
            
            miku_sphere = GameObject('miku', position=[0, 0.02, 0], scale=0.065)
            
            miku_corrmap = CorrespondMap.Load(TEMP_DIR / 'test_corresponder_finished' / 'test_15', name='miku corrmap')
            #miku_corrmap = CorrespondMap.Load(TEMP_DIR / 'test_corresponder_finished' / 'test_17', name='miku corrmap')
            miku_corrmap.k = 3
            
            corrmap_material = Material.DefaultTransparentMaterial()
            corrmap_material.addDefaultTexture(miku_corrmap, DefaultTextureType.CorrespondMap)
            
            miku_sphere.addComponent(CorrMapRenderer, corrmaps=miku_corrmap, materials=[corrmap_material,], use_texcoord_id=True)
            miku_sphere.addComponent(AutoRotation)
            miku_sphere.addComponent(SpriteInfo, auto_spriteID=True, prompt='')
            
    Sample.Run(winSize=(512, 512),
               mode = EngineMode.GAME,
               mapSavingInterval=1,
               needOutputMaps=False,
               disableComfyUI=True)
