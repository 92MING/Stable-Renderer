import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Mesh, Material, Material_MTL, Key, MouseButton, Color
from utils.path_utils import *

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.5)
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
            mikuPath = os.path.join(RESOURCES_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'))
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials

            camera = GameObject('Camera')
            camera.addComponent(Camera)
            camera.addComponent(CameraControl)

            miku = GameObject('miku', position=[0, 0, 0], scale=[0.1, 0.1, 0.1])
            meshRenderer:MeshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(AutoRotation)

            plane = GameObject('plane', position=[0, 0, 0], scale=[1.5, 1.5, 1.5])
            meshRenderer:MeshRenderer = plane.addComponent(MeshRenderer, mesh=Mesh.Plane())
            meshRenderer.addMaterial(Material.Debug_Material(whiteMode=True))

    Sample.Run(enableGammaCorrection=False,
               enableHDR=False,
               debug=True,
               winSize=(512, 512),
               needOutputMaps=False,)
