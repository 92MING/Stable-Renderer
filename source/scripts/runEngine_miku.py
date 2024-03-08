import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController
from engine.static import Mesh, Material, Material_MTL, GLFW_Key, MouseButton
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
            camera.addComponent(CameraController, defaultPos=[1.3, 2.8, 1.3], defaultLookAt=[0, 2.8, 0])

            miku = GameObject('miku', position=[0, 0, 0], scale=[0.16, 0.16, 0.16])
            meshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
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
               mapSavingInterval=4,
               needOutputMaps=True,)
