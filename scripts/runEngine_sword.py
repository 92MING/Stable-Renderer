import torch
import math
torch.cuda.current_device()

import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

import glm
import os.path
from engine.runtime.components import Camera, MeshRenderer
from engine.runtime.gameObj import GameObject
from engine.runtime.component import Component
from engine.engine import Engine
from engine.runtime.components import CameraController, HelicalOrbit
from engine.static import Material, Mesh, Texture, DefaultTextureType, GLFW_Key
from common_utils.path_utils import EXAMPLE_3D_MODEL_DIR
from common_utils.spherical_cache import ViewPoint

import json
from pydantic_core import to_jsonable_python


if __name__ == '__main__':
    class ControlSword(Component):
        '''Control the sword with W, A, S, D keys. The boat will move forward, backward, left, right.'''
        
        deceleration_rate: float = 0.98
        angular_deceleration_rate: float = 0.95
        
        acceleration: float = 0.05
        angular_acceleration: float = 5
        max_spd = 0.5
        max_angular_spd = 90
        
        velocity: glm.vec3 = glm.vec3(0)
        angular_velocity: float = 0
        
        _inputManager = None
        @property
        def inputManager(self):
            if self._inputManager is None:
                self._inputManager = self.engine.InputManager
            return self._inputManager
        
        _runtimeManager = None
        @property
        def runtimeManager(self):
            if self._runtimeManager is None:
                self._runtimeManager = self.engine.RuntimeManager
            return self._runtimeManager
        
        def update_physics(self):
            if glm.length(self.velocity) > 0.005:
                self.velocity *= self.deceleration_rate
                self.transform.position += (self.transform.forward + self.velocity) * self.runtimeManager.DeltaTime
            else:
                self.velocity = glm.vec3(0)
                
            if abs(self.angular_velocity) > 0.005:
                self.angular_velocity *= self.angular_deceleration_rate
                self.transform.rotateLocalY(self.angular_velocity * self.runtimeManager.DeltaTime)
            else:
                self.angular_velocity = 0
        
        def fixedUpdate(self):
            if self.inputManager.GetKey(GLFW_Key.W):
                self.velocity += self.transform.forward * self.acceleration
                
            if self.inputManager.GetKey(GLFW_Key.S):
                self.velocity -= self.transform.forward * self.acceleration
                
            if self.inputManager.GetKey(GLFW_Key.A):
                self.angular_velocity += self.angular_acceleration
                
            if self.inputManager.GetKey(GLFW_Key.D):
                self.angular_velocity -= self.angular_acceleration
                
            self.velocity = glm.clamp(self.velocity, -self.max_spd, self.max_spd)
            self.angular_velocity = -self.max_angular_spd if self.angular_velocity < -self.max_angular_spd else self.max_angular_spd if self.angular_velocity > self.max_angular_spd else self.angular_velocity
            
            self.update_physics()

    class HelicalOrbitWrapper(HelicalOrbit):
        historical_pos = []
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def update(self):
            super().update()
            view_point = ViewPoint.from_cartesian(*self.transform.position)
            self.historical_pos.append(view_point)
    
    class Sample(Engine):
        def beforePrepare(self):

            self.swordMesh = Mesh.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'sword', 'sword.obj'))
            self.swordMaterial = Material.DefaultOpaqueMaterial()

            sword_diffuse_tex = Texture.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'sword', 'Sting_Base_Color.png'))
            self.swordMaterial.addDefaultTexture(sword_diffuse_tex, DefaultTextureType.DiffuseTex)
            # self.swordMaterial.addDefaultTexture(Texture.Load(os.path.join(EXAMPLE_3D_MODEL_DIR, 'sword', 'Sting_Normal.png')), DefaultTextureType.NormalTex)
            # self.swordMaterial.addDefaultTexture(Texture.CreateNoiseTex(), DefaultTextureType.NoiseTex)
            
            self.sword = GameObject('Sword', position=[0, 0, 0])
            self.sword.addComponent(MeshRenderer, mesh=self.swordMesh, materials=self.swordMaterial)
            
            initial_position = [0, 0, -6]
            self.camera = GameObject('Camera', position=initial_position)
            self.camera.addComponent(Camera)
            self.camera.addComponent(CameraController, defaultPos=initial_position, defaultLookAt=[0, 0, 0])
            self.camera.addComponent(HelicalOrbitWrapper, radius=10, theta_speed=1, phi=45)


    Sample.Run(enableGammaCorrection=True,
            debug=False,
            winSize=(512, 512),
            mapSavingInternal=1,
            needOutputMaps=False,
            disableComfyUI=True,
            fixedUpdateMaxFPS=60,)
    
        
