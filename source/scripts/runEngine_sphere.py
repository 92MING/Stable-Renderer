import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)
import glm
from engine.runtime.components import Camera, MeshRenderer, CameraController, HelicalOrbit
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.static import Material, Mesh, Texture, DefaultTextureType, TextureWrap
from engine.managers import DiffusionManager
from engine.static.enums import TextureWrap

from common_utils.path_utils import *
from common_utils.data_struct.view_point import ViewPoint

from pydantic_core import to_jsonable_python
import json

if __name__ == '__main__':

    class HelicalOrbitWrapper(HelicalOrbit):
        historical_pos = []
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def update(self):
            super().update()
            view_point = ViewPoint.from_cartesian(*self.transform.position)
            self.historical_pos.append(view_point)

    class Sample(Engine):
        def beforePrepare(self):
            self.sphere_mesh = Mesh.Sphere()
            self.sphere_mat = Material.Default_Opaque_Material()
            self.sphere_mat.addDefaultTexture(
                Texture.Load(
                    os.path.join(EXAMPLE_3D_MODEL_DIR, 'debug', 'debug_uv_texture.jpg'),
                    s_wrap=TextureWrap.CLAMP_TO_EDGE,
                    ),
                    DefaultTextureType.DiffuseTex
            )

            initial_position = [0, 0, -6]
            self.camera = GameObject('Main Cam', position=initial_position)
            self.camera.addComponent(Camera)
            # self.camera.addComponent(CameraController, defaultPos=initial_position, defaultLookAt=[0, 0, 0])
            self.camera.addComponent(HelicalOrbitWrapper, theta_speed=1, phi=180)

            self.sphere = GameObject('Sphere', position=[0, 0, 0], scale=[3, 3, 3])
            self.sphere.addComponent(MeshRenderer, mesh=self.sphere_mesh, materials=self.sphere_mat)
            

    Sample.Run(enableGammaCorrection=False,
               debug=False,
               mapSavingInterval=1,
               winSize=(512, 512),
               needOutputMaps=True,
               startComfyUI=False)

    with open('sphere_historical_pos.json', 'w') as f:
        json.dump(HelicalOrbitWrapper.historical_pos, f, default=to_jsonable_python)
        print(len(HelicalOrbitWrapper.historical_pos))