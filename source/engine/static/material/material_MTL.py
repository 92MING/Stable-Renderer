import os.path
from .material import Material, DefaultTextureType
from typing import Tuple, Dict, TYPE_CHECKING, Optional
from ..texture import Texture
from common_utils.global_utils import GetGlobalValue
if TYPE_CHECKING:
    from engine.engine import Engine

class Material_MTL(Material):
    '''Special Material class for loading mtl file. It is used for loading mtl file only.'''

    _Format = 'mtl'

    _realName: Optional[str] = None
    '''Real name in .mtl file'''

    @property
    def realName(self):
        '''the name specified in "newmtl" line in mtl file'''
        return self._realName

    def _getTex(self, path, name):
        tex = Texture.Find(name.split('.')[0])
        if tex is None:
            tex = Texture.Load(path=path)
        return tex

    def load(self, dirPath:str, dataLines:Tuple[str,...]):
        '''
        Different to super().load(self, path), this Material_MTL will load the data from a list of strings(lines) directly.
        The "dirPath" is the folder path of the mtl file. It is for searching the texture files.
        This function should be used as internal method only.
        '''
        for line in dataLines:
            if line.startswith('#'): continue
            elif line.startswith('Ns'):
                # TODO: specular exponent
                pass
            elif line.startswith('Ka'):
                # TODO: ambient color
                pass
            elif line.startswith('Kd'):
                # TODO: diffuse color
                pass
            elif line.startswith('Ks'):
                # TODO: specular color
                pass
            elif line.startswith('Ni'):
                # TODO: optical density
                pass
            elif line.startswith('d'):
                # TODO: dissolve
                pass
            elif line.startswith('illum'):
                # TODO: illumination method
                pass
            elif line.startswith('map_Kd'):
                name = line.split(' ')[1] # texture file name, e.g. "texture.png"
                path = os.path.join(dirPath, name)
                if os.path.exists(path):
                    self.addDefaultTexture(self._getTex(path, name), DefaultTextureType.DiffuseTex)
            elif line.startswith('map_Ks'):
                name = line.split(' ')[1]
                path = os.path.join(dirPath, name)
                if os.path.exists(path):
                    self.addDefaultTexture(self._getTex(path, name), DefaultTextureType.SpecularTex)
            elif line.startswith('map_Ns'):
                # TODO: specular highlight
                pass
            elif line.startswith('map_d'):
                name = line.split(' ')[1]
                path = os.path.join(dirPath, name)
                if os.path.exists(path):
                    self.addDefaultTexture(self._getTex(path, name), DefaultTextureType.AlphaTex)
            elif line.startswith('map_bump'):
                name = line.split(' ')[1]
                path = os.path.join(dirPath, name)
                if os.path.exists(path):
                    self.addDefaultTexture(self._getTex(path, name), DefaultTextureType.NormalTex)

    @classmethod
    def Load(cls, path) -> Tuple['Material_MTL']:
        '''
        Load a .mtl file and return a tuple of Material_MTL objects.

        Args:
            path: path of the .mtl file

        Returns:
            tuple of Material_MTL objects(since a .mtl file has multiple materials)
        '''
        path, _ = cls._GetPathAndName(path)
        dirPath = os.path.dirname(path)

        with open(path, 'r') as f:
            lines = [line.strip('\n') for line in f.readlines()]
            materials = []
            currMatDataLines = []
            currMat: Optional[Material_MTL] = None
            engine: 'Engine' = GetGlobalValue('_ENGINE_SINGLETON')

            for line in lines:
                if line.startswith('#') or line in ("\n", ""): continue
                elif line.startswith('newmtl'):

                    # save the previous material
                    if currMat is not None:
                        currMat.load(dirPath, tuple(currMatDataLines))
                        currMatDataLines.clear()

                    # find a proper name for the new material
                    realMatName = line.split(' ')[1]
                    matName = realMatName
                    count = 0
                    while matName in Material.AllInstances():
                        count += 1
                        matName = f'{matName}_{count}'

                    currMat = cls.Default_Opaque_Material(name=matName) if not engine.IsDebugMode else cls.Debug_Material(name=matName)
                    currMat._realName = realMatName
                    materials.append(currMat)

                elif currMat is not None:
                    currMatDataLines.append(line)

            # save the last material
            if currMat is not None:
                currMat.load(dirPath, tuple(currMatDataLines))

            return tuple(materials)



__all__ = ['Material_MTL']