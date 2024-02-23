import sys, os
sys.path.append(os.getcwd())

import os.path
from runtime.components import Camera, MeshRenderer, CameraControl
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture, DefaultTextureType
from utils.path_utils import *


if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.5)

    class Sample(Engine):
        def beforePrepare(self):
            self.ballMesh = Mesh.Load(os.path.join(RESOURCES_DIR, 'basketball', 'ball.obj'))
            self.ballMaterial = Material.Default_Opaque_Material()
            self.ballMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'basketball', 'ball_BaseColor_512.png')), DefaultTextureType.DiffuseTex)

            # uncomment the following line to add normal map
            # self.ballMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'basketball', 'ball_Normal.png')), DefaultTextureType.NormalTex)

            self.camera = GameObject('Camera', position=[4, 8, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraControl, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])

            self.ball = GameObject('ball', position=[0, 0, 0])
            self.ball.addComponent(MeshRenderer, mesh=self.ballMesh, materials=self.ballMaterial)
            self.ball.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=True,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=True,)
