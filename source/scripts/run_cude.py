import sys, os
sys.path.append(os.getcwd())

import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture, DefaultTextureType, Key, MouseButton
from utils.path_utils import *


if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.2)

    class CameraControl(Component):
        def awake(self):
            self.defaultPos = [0, 1.25, 3]
            self.moveSpd = 0.005
            self.moveFowardSpd = 0.1
            self._camera = None
        def start(self):
            self.transform.localPos = self.defaultPos
            self.transform.lookAt([0, 1, 0])
            self.transform.setLocalRotY(0)
        @property
        def camera(self)->Camera:
            if self._camera is None:
                self._camera = self.gameObj.getComponent(Camera)
            return self._camera
        def update(self):
            inputManager = self.engine.InputManager
            if inputManager.GetKeyDown(Key.R):
                self.transform.globalPos = self.defaultPos
            if inputManager.GetMouseBtn(MouseButton.LEFT):
                up = self.transform.up
                right = self.transform.right
                mouseDelta = inputManager.MouseDelta
                self.transform.globalPos = self.transform.globalPos + right * -mouseDelta[0] * self.moveSpd + up * mouseDelta[1] * self.moveSpd
            if inputManager.HasMouseScrolled:
                self.transform.globalPos = self.transform.globalPos + self.transform.forward * inputManager.MouseScroll[1] * self.moveFowardSpd

    class Sample(Engine):
        def beforePrepare(self):
            self.cube_mesh = Mesh.Cube()
            self.cube_mat = Material.Default_Opaque_Material()
            self.cube_mat.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')), DefaultTextureType.DiffuseTex)
            # self.boatMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatNormal.png')), DefaultTextureType.NormalTex)

            self.camera = GameObject('Camera', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraControl)
            # self.camera.transform.lookAt([0, 0, 0])

            self.cube = GameObject('Cube', position=[0, 0, 0])
            self.cube.addComponent(MeshRenderer, mesh=self.cube_mesh, materials=self.cube_mat)
            self.cube.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=True,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=True,)
