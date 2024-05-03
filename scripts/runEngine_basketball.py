import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

import os.path
from engine.runtime.components import Camera, MeshRenderer, CameraController
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.static import Material, Mesh, Texture, DefaultTextureType
from common_utils.path_utils import *


if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.5)

    class Sample(Engine):
        def beforePrepare(self):
            self.ballMesh = Mesh.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'basketball', 'ball.obj'))
            self.ballMaterial = Material.Default_Opaque_Material()
            self.ballMaterial.addDefaultTexture(Texture.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'basketball', 'ball_BaseColor_512.png')), DefaultTextureType.DiffuseTex)

            # uncomment the following line to add normal map
            # self.ballMaterial.addDefaultTexture(Texture.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'basketball', 'ball_Normal.png')), DefaultTextureType.NormalTex)

            self.camera = GameObject('Camera', position=[4, 8, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraController, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])

            self.ball = GameObject('ball', position=[0, 0, 0])
            self.ball.addComponent(MeshRenderer, mesh=self.ballMesh, materials=self.ballMaterial)
            self.ball.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=True,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=True,)
