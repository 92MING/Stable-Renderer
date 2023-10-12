from static.resourcesObj import ResourcesObj
import os
import OpenGL.GL as gl
from PIL import Image

class Texture(ResourcesObj):
    _BaseName = 'Texture'

    def load(self, path):
        '''Default load by PIL'''
        image = Image.open(path)
        converted = image.convert(mode)
        self.buffer = converted.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        self.height = image.height
        self.width = image.width
        self.format = mode
        len(self.buffer) / (image.width * image.height)
        image.close()
        self.textureGLID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureGLID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.buffer)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
    def sendToGPU(self):
        '''Send data to GPU memory'''
        raise NotImplementedError
    def clear(self):
        '''Clear all data and release GPU memory if it has been loaded'''
        raise NotImplementedError

    @classmethod
    def Load(cls, path, name=None):
        path, name = cls._GetPathAndName(path, name)
        if '.' in os.path.basename(path):
            ext = path.split('.')[-1]
            formatCls = cls.FindFormat(ext)
            if formatCls is None:
                raise ValueError(f'unsupported texture format: {ext}')
            texture = formatCls(name)
            texture.load(path)
        else: # default load by PIL
            texture = cls(name)
            texture.load(path)
        return texture
