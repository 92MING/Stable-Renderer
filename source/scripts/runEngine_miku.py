import sys, os
sys.path.append(os.getcwd())

import os.path
from runtime.components import Camera, MeshRenderer
from runtime.gameObj import GameObject
from runtime.component import Component
from runtime.engine import Engine
from static import Mesh, Material, Shader, Material_MTL
from utils.path_utils import *

if __name__ == '__main__':

    class AutoRotation(Component):
        def update(self):
            self.transform.rotateLocalY(0.2)

    class Sample(Engine):
        def beforePrepare(self):
            mikuPath = os.path.join(RESOURCES_DIR, 'miku')
            mikuMesh = Mesh.Load(os.path.join(mikuPath, 'miku.obj'))
            mikuMaterials = Material_MTL.Load(os.path.join(mikuPath, 'miku.mtl')) # dict of materials

            camera = GameObject('Camera', position=[60, 20, -25])
            camera.addComponent(Camera)
            camera.transform.lookAt([-70, 25, 0])

            miku = GameObject('miku', position=[0, 0, 0], scale=[1, 1, 1])
            meshRenderer:MeshRenderer = miku.addComponent(MeshRenderer, mesh=mikuMesh)
            meshRenderer.load_MTL_Materials(mikuMaterials)
            miku.addComponent(AutoRotation)

    Sample.Run(enableGammaCorrection=False,
               enableHDR=False,
               debug=False,
               winSize=(512, 512),
               needOutputMaps=True,)
