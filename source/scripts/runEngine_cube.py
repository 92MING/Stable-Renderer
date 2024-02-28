import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

from utils.path_utils import RESOURCES_DIR
from engine.runtime.components import Camera, MeshRenderer, CameraController
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.static import Material, Mesh, Texture, DefaultTextureType, TextureWrap
from engine.managers import DiffusionManager

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.3)

    class Sample(Engine):
        def beforePrepare(self):
            self.cube_mesh = Mesh.Cube()
            self.cube_mat = Material.Default_Opaque_Material()

            tex_path = os.path.join(RESOURCES_DIR, 'boat', 'boatColor.png')
            tex = Texture.Load(tex_path, s_wrap=TextureWrap.CLAMP_TO_BORDER, t_wrap=TextureWrap.CLAMP_TO_BORDER)
            self.cube_mat.addDefaultTexture(tex, DefaultTextureType.DiffuseTex)

            self.camera = GameObject('Main Cam', position=[4, 4, -3])
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraController, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])

            self.cube = GameObject('Rect Cube', position=[0, 0, 0], scale=[0.9, 0.9 ,2.6])
            self.cube.addComponent(MeshRenderer, mesh=self.cube_mesh, materials=self.cube_mat)
            self.cube.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=False,
               debug=False,
               mapSavingInterval=8,
               winSize=(512, 512),
               needOutputMaps=False)