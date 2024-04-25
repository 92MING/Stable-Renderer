import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

from engine.runtime.components import Camera, HelicalOrbit, BakeRenderer
from engine.runtime.gameObj import GameObject
from engine.engine import Engine
from engine.static import Material, Mesh, Texture, DefaultTextureType, TextureWrap
from common_utils.path_utils import *


if __name__ == '__main__':

    class Sample(Engine):
        def beforePrepare(self):
            sphere_mesh = Mesh.Sphere()
            sphere_mat = Material.Default_Opaque_Material()
            noise_tex = Texture.CreateNoiseTex(t_wrap=TextureWrap.CLAMP_TO_EDGE, s_wrap=TextureWrap.CLAMP_TO_EDGE)
            sphere_mat.addDefaultTexture(noise_tex,
                                         DefaultTextureType.NoiseTex,)
            sphere_mat.addDefaultTexture(
                Texture.Load(
                    os.path.join(EXAMPLE_3D_MODEL_DIR, 'debug', 'debug_uv_texture.jpg'),
                    s_wrap=TextureWrap.CLAMP_TO_EDGE,
                ),
                DefaultTextureType.DiffuseTex
            )
            sphere = GameObject('Sphere', position=[0, 0, 0], scale=[3, 3, 3])
            sphere.addComponent(BakeRenderer, mesh=sphere_mesh, materials=sphere_mat)

            initial_position = [0, 0, -6]
            camera = GameObject('Main Cam', position=initial_position)
            camera.addComponent(Camera)
            self.helical_orbit_wrapper = camera.addComponent(HelicalOrbit, theta_speed=1, phi=180)

    Sample.Run(enableGammaCorrection=False,
               debug=False,
               mapSavingInterval=1,
               winSize=(512, 512),
               needOutputMaps=False,
               startComfyUI=False)