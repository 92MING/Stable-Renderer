import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture
from utils.path_utils import RESOURCES_DIR

if __name__ == '__main__':

    class Boat(Component):
        def update(self):
            self.transform.rotateLocalY(0.1)

    class Sample(Engine):
        def beforePrepare(self):
            self.boatMesh = Mesh.Load(os.path.join(RESOURCES_DIR, 'boat', 'boat.obj'))
            #self.boatMaterial = Material.Debug_Material()
            self.boatMaterial = Material.Default_Opaque_Material()
            self.boatMaterial.addDiffuseMap(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')))
            self.boatMaterial.addNormalMap(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatNormal.png')))

            self.camera = GameObject('Camera', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.transform.lookAt([0, 0, 0])

            self.boat = GameObject('Boat', position=[0, 0, 0])
            self.boat.addComponent(MeshRenderer, mesh=self.boatMesh, material=self.boatMaterial)
            self.boat.addComponent(Boat)

    Sample.Run()