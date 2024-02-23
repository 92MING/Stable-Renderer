import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime.components import Camera, MeshRenderer, CameraControl
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture, DefaultTextureType

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.3)

    class Sample(Engine):
        def beforePrepare(self):
            self.cube_mesh = Mesh.Cube()
            self.cube_mat = Material.Default_Opaque_Material()
            self.cube_mat.addDefaultTexture(Texture.CreateVirtualTex(), DefaultTextureType.DiffuseTex)

            self.camera = GameObject('Main Cam', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraControl, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])

            self.cube = GameObject('Rect Cube', position=[0, 0, 0], scale=[0.9, 0.9 ,2.6])
            self.cube.addComponent(MeshRenderer, mesh=self.cube_mesh, materials=self.cube_mat)
            self.cube.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=False,
               debug=False,
               mapSavingInterval=8,
               winSize=(512, 512),
               needOutputMaps=False,)
