from static.resourcesObj import ResourcesObj
import os
import OpenGL.GL as gl
from PIL import Image

class Texture(ResourcesObj):
    _BaseName = 'Texture'

    def load(self, path):
        '''Default load by PIL'''
        pass
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
        else:
            texture = cls(name)
            texture.load(path)
        return texture

class Texture_DSS(Texture):
    _Format = 'dds'

    def load(self, path):
        dds = DDSImage(fname)
        ddsBuffer = dds.data()
        self.textureGLID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureGLID)
        format = dds.format().getOpenGLformat()
        for level in range(0, dds.mip_count()):
            mipMap = dds.mip_map(level)
            width, height, offset, size = mipMap.width, mipMap.height, mipMap.offset, mipMap.size
            if dds.compressed():
                glCompressedTexImage2D(GL_TEXTURE_2D, level, format, width, height, 0, size,
                                       ddsBuffer[offset:offset + size])
            else:
                glTexImage2D(GL_TEXTURE_2D, level, format, width, height, 0, format, GL_UNSIGNED_BYTE,
                             ddsBuffer[offset:offset + size])
        self.inversedVCoords = True