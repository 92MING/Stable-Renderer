import os.path
import glm
from runtime.engine import Engine
from static.texture import Texture
from static.shader import Shader
from static.mesh import Mesh
from utils.path_utils import RESOURCES_DIR
import OpenGL.GL as gl

if __name__ == '__main__':
    class Sample(Engine):
        def beforePrepare(self):
            boatDir = os.path.join(RESOURCES_DIR, 'boat')
            self.boat = Mesh.Load(os.path.join(boatDir, 'boat.obj'))
            self.boatShader = Shader('boat_shader',os.path.join(boatDir, 'boat_vs.glsl'), os.path.join(boatDir, 'boat_fs.glsl'))
            self.boatDiffuseTex = Texture.Load(os.path.join(boatDir, 'boatColor.png'), 'boat_diffuse')
            proj = glm.perspective(glm.radians(45.0), self.WindowManager.AspectRatio, 0.1, 1000.0)
            view = glm.lookAt(glm.vec3(4, 4, -3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
            self.RenderManager.UpdateUBO_ProjMatrix(proj)
            self.RenderManager.UpdateUBO_ViewMatrix(view)

        def beforeFrameRun(self):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            modelM = self.RenderManager.UBO_ModelMatrix
            modelM = glm.rotate(modelM, glm.radians(0.05), glm.vec3(0.0, 1.0, 0.0))
            self.RenderManager.UpdateUBO_ModelMatrix(modelM)

            self.boatShader.useProgram()
            self.boatDiffuseTex.bind(0, self.boatShader.getUniformID('boatDiffuseTex'))
            self.boat.draw()

    Sample().run()