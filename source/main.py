import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Material, Mesh, Texture
from utils.path_utils import RESOURCES_DIR
import shutil

if __name__ == '__main__':
    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.2)

    class Sample(Engine):
        def beforePrepare(self):
            self.boatMesh = Mesh.Load(os.path.join(RESOURCES_DIR, 'boat', 'boat.obj'))
            self.boatMaterial = Material.Default_Opaque_Material()
            self.boatMaterial.addDiffuseMap(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')))
            self.boatMaterial.addNormalMap(Texture.Load(os.path.join(RESOURCES_DIR, 'boat', 'boatNormal.png')))

            self.camera = GameObject('Camera', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.transform.lookAt([0, 0, 0])

            self.boat = GameObject('Boat', position=[0, 0, 0])
            self.boat.addComponent(MeshRenderer, mesh=self.boatMesh, material=self.boatMaterial)
            self.boat.addComponent(AutoRotation)

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    Sample.Run(enableGammaCorrection=True,
               debug=True,
               output_dir='./tmp',
               output_subfolders=['color', 'pos', 'normal', 'id', 'depth'],
               save_map_per_num_frame=30)