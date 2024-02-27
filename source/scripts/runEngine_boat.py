import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

import glm
import os.path
from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraControl
from engine.static import Material, Mesh, Texture, DefaultTextureType, Key, MouseButton
from utils.path_utils import *


if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.2)
            
    class ControlBoat(Component):
        def update(self):
            inputManager = self.engine.InputManager
            direction = self.transform.forward
            if inputManager.GetKey(Key.W):
                self.transform.position = self.transform.position + self.transform.forward * 0.1
                direction = direction + self.transform.forward * 0.1
            if inputManager.GetKey(Key.S):
                self.transform.position = self.transform.position - self.transform.forward * 0.1
                direction = direction - self.transform.forward * 0.1
            if inputManager.GetKey(Key.A):
                self.transform.position = self.transform.position - self.transform.right * 0.1
                direction = direction - self.transform.right * 0.1
            if inputManager.GetKey(Key.D):
                self.transform.position = self.transform.position + self.transform.right * 0.1
                direction = direction + self.transform.right * 0.1
                
            self.transform.forward = glm.normalize(direction)

            
    class Sample(Engine):
        def beforePrepare(self):
            self.boatMesh = Mesh.Load(os.path.join(RESOURCES_DIR, 'boat', 'boat.obj'))
            self.boatMaterial = Material.Default_Opaque_Material()
            self.boatMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')), DefaultTextureType.DiffuseTex)

            # uncomment this line to enable normal map
            # self.boatMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatNormal.png')), DefaultTextureType.NormalTex)

            self.boat = GameObject('Boat', position=[0, 0, 0])
            self.boat.addComponent(MeshRenderer, mesh=self.boatMesh, materials=self.boatMaterial)
            # self.boat.addComponent(AutoRotation)
            self.boat.addComponent(ControlBoat)
            
            self.camera = GameObject('Camera', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraControl, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])


    Sample.Run(enableGammaCorrection=True,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=False,)
