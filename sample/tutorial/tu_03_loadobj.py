# import os,sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from OpenGL.GL import *  # pylint: disable=W0614

import glm
from utils.glutWindow import GlutWindow
from utils.objLoader import objLoader
from utils.shaderLoader import Shader
from utils.textureLoader import textureLoader


class Tu01Win(GlutWindow):

	class GLContext(object):
		pass
	def init_opengl(self):
		glClearColor(0.0,0,0.4,0)
		glDepthFunc(GL_LESS)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_CULL_FACE)

	def init_context(self):
		self.context = self.GLContext()



		self.shader = shader = Shader()
		shader.initShaderFromGLSL(["glsl/tu02/vertex.glsl"],["glsl/tu02/fragment.glsl"])

		self.context.MVP_ID   = glGetUniformLocation(shader.program,"MVP")
		self.context.TextureID =  glGetUniformLocation(shader.program, "myTextureSampler")



		texture = textureLoader("resources/tu03/uvmap.dds")
		#texture = textureLoader("opengl_tutorial/models/tu02/uvtemplate.dds")

		self.context.textureGLID = texture.textureGLID


		model = objLoader("resources/tu03/cube.obj").to_array_style()
		self.context.vertexbuffer  = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER,self.context.vertexbuffer)
		glBufferData(GL_ARRAY_BUFFER,len(model.vertexs)*4,(GLfloat * len(model.vertexs))(*model.vertexs),GL_STATIC_DRAW)

		if(texture.inversedVCoords):
			for index in range(0,len(model.texcoords)):
				if(index % 2):
					model.texcoords[index] = 1.0 - model.texcoords[index]

		self.context.uvbuffer  = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER,self.context.uvbuffer)
		glBufferData(GL_ARRAY_BUFFER,len(model.texcoords)*4,(GLfloat * len(model.texcoords))(*model.texcoords),GL_STATIC_DRAW)

	def calc_MVP(self,width=1920,height=1080):

		self.context.Projection = glm.perspective(glm.radians(45.0),float(width)/float(height),0.1,1000.0)
		self.context.View =  glm.lookAt(glm.vec3(4,3,-3), # Camera is at (4,3,-3), in World Space
						glm.vec3(0,0,0), #and looks at the (0.0.0))
						glm.vec3(0,1,0) ) #Head is up (set to 0,-1,0 to look upside-down)
		#fixed Cube Size
		self.context.Model=  glm.mat4(1.0)
		#print self.context.Model
		self.context.MVP =  self.context.Projection * self.context.View * self.context.Model	

	def resize(self,Width,Height):
		
		glViewport(0, 0, Width, Height)
		self.calc_MVP(Width,Height)

	def ogl_draw(self):

		print "draw++"
		#print self.context.MVP
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		self.shader.begin()
		glUniformMatrix4fv(self.context.MVP_ID,1,GL_FALSE,glm.value_ptr(self.context.MVP))


		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.context.textureGLID)
		glUniform1i(self.context.TextureID, 0)


		glEnableVertexAttribArray(0)
		glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

		glEnableVertexAttribArray(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.context.uvbuffer)
		glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,0,None)


		glDrawArrays(GL_TRIANGLES, 0, 12*3) # 12*3 indices starting at 0 -> 12 triangles

		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(1)
		self.shader.end()
        

if __name__ == "__main__":

    win = Tu01Win()
    win.init_opengl()
    win.init_context()
    win.run()
