import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture, DefaultTextureType
from utils.path_utils import *


if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.2)

    class Sample(Engine):
        def beforePrepare(self):
            self.boatMesh = Mesh.Load(os.path.join(RESOURCES_DIR, 'boat', 'boat.obj'))
            self.boatMaterial = Material.Default_Opaque_Material()
            self.boatMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')), DefaultTextureType.DiffuseTex)
            self.boatMaterial.addDefaultTexture(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatNormal.png')), DefaultTextureType.NormalTex)

            self.camera = GameObject('Camera', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.transform.lookAt([0, 0, 0])

            self.boat = GameObject('Boat', position=[0, 0, 0])
            self.boat.addComponent(MeshRenderer, mesh=self.boatMesh, materials=self.boatMaterial)
            self.boat.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=True,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=False,)
