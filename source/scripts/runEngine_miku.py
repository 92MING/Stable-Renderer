import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from runtime.components import CameraControl
from static import Mesh, Material, Material_MTL, Key, MouseButton
from utils.path_utils import *

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.5)

    class Sample(Engine):
        def beforePrepare(self):
            mikuPath = os.path.join(RESOURCES_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'))
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials

            camera = GameObject('Camera')
            camera.addComponent(Camera)
            camera.addComponent(CameraControl, defaultPos=[4, 4, -3], defaultLookAt=[0, 0, 0])

            miku = GameObject('miku', position=[0, 0, 0], scale=[0.1, 0.1, 0.1])
            meshRenderer:MeshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(AutoRotation)

            # uncomment to add a plane
            #plane = GameObject('plane', position=[0, 0, 0], scale=[1.5, 1.5, 1.5])
            #meshRenderer:MeshRenderer = plane.addComponent(MeshRenderer, mesh=Mesh.Plane())
            #meshRenderer.addMaterial(Material.Debug_Material(whiteMode=True))

    Sample.Run(enableGammaCorrection=False,
               enableHDR=False,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=True,)
