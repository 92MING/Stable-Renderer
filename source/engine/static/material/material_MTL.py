import os

from attr import attrib, attrs
from typing import Tuple, Dict, TYPE_CHECKING, Optional, List, Union
from common_utils.global_utils import GetGlobalValue
from .material import Material, DefaultTextureType
from ..texture import Texture

if TYPE_CHECKING:
    from engine.engine import Engine

@attrs(eq=False, repr=False)
class Material_MTL(Material):
    '''Special Material class for loading mtl file. It is used for loading mtl file only.'''

    Format = 'mtl'

    real_name: Optional[str] = attrib(default=None)
    '''Real name in .mtl file'''
    ka: Tuple[float, float, float] = attrib(default=(0.0, 0.0, 0.0))
    '''Ambient color'''
    kd: Tuple[float, float, float] = attrib(default=(0.0, 0.0, 0.0))
    '''Diffuse color'''
    ks: Tuple[float, float, float] = attrib(default=(0.0, 0.0, 0.0))
    '''Specular color'''
    ns: float = attrib(default=0.0)
    '''Specular exponent'''
    ni: float = attrib(default=0.0)
    '''Optical density'''
    d: float = attrib(default=1.0)
    '''Dissolve'''
    illum: int = attrib(default=0)
    '''Illumination method'''

    @classmethod
    def _Parse(cls, 
               dataLines: Union[Tuple[str, ...], List[str]],
               data_paths: List[str], 
               real_name: Optional[str] = None,
               add_textures=True,
               is_debug_mode=False):
        data = {}
        textures = []   # [(tex, type), ...]
        
        def _search_tex(filename):
            alias = filename.split('.')[0]
            for path in data_paths:
                if os.path.exists(os.path.join(path, filename)):
                    return Texture.Load(os.path.join(path, filename), alias=alias)
            return None
        
        for line in dataLines:
            if line.startswith('#'): continue
            elif line.startswith('Ns'):
                data['ns'] = float(line.split(' ')[1])
            elif line.startswith('Ka'):
                data['ka'] = tuple(map(float, line.split(' ')[1:]))
            elif line.startswith('Kd'):
                data['kd'] = tuple(map(float, line.split(' ')[1:]))
            elif line.startswith('Ks'):
                data['ks'] = tuple(map(float, line.split(' ')[1:]))
            elif line.startswith('Ni'):
                data['ni'] = float(line.split(' ')[1])
            elif line.startswith('d'):
                data['d'] = float(line.split(' ')[1])
            elif line.startswith('illum'):
                data['illum'] = int(line.split(' ')[1])
            elif add_textures and line.startswith('map_Kd'):
                name = line.split(' ')[1] # texture file name, e.g. "texture.png"
                if tex:=_search_tex(name):
                    textures.append((tex, DefaultTextureType.DiffuseTex))
            elif add_textures and line.startswith('map_Ks'):
                name = line.split(' ')[1]
                if tex:=_search_tex(name):
                    textures.append((tex, DefaultTextureType.SpecularTex))
            elif add_textures and line.startswith('map_d'):
                name = line.split(' ')[1]
                if tex:=_search_tex(name):
                    textures.append((tex, DefaultTextureType.AlphaTex))
            elif add_textures and line.startswith('map_bump'):
                name = line.split(' ')[1]
                if tex:=_search_tex(name):
                    textures.append((tex, DefaultTextureType.NormalTex))
        if is_debug_mode:
            mat = cls.DefaultDebugMaterial(**data, real_name=real_name)
        else:
            mat = cls.DefaultOpaqueMaterial(**data, real_name=real_name)
        for tex, texType in textures:
            mat.addDefaultTexture(tex, texType)
        return mat

    @classmethod
    def Load(cls, 
             path: str, 
             extra_paths: Union[List[str], Tuple[str, ...], str, None] = None,
             add_textures = True) -> Tuple['Material_MTL']:
        '''
        Load a .mtl file and return a tuple of Material_MTL objects.

        Args:
            - path: path of the .mtl file
            - extra_paths: extra paths for searching textures (in case `add_textures` is True)
            - add_textures: whether to add textures to the material
        Returns:
            tuple of Material_MTL objects(since a .mtl file has multiple materials)
        '''
        
        if isinstance(extra_paths, str):
            extra_paths = [extra_paths]
        extra_paths = extra_paths or []
        
        dirPath = os.path.dirname(path)
        
        with open(path, 'r') as f:
            lines = [line.strip('\n') for line in f.readlines()]
            materials = []
            current_data_lines = []
            all_paths = [dirPath] + list(extra_paths)
            real_name: str = None
            
            engine: 'Engine' = GetGlobalValue('__ENGINE_INSTANCE__', None)  # type: ignore
            if not engine:
                debug_mode = False
            else:
                debug_mode = engine.IsDebugMode

            for line in lines:
                if line.startswith('#') or line in ("\n", ""): 
                    continue
                elif line.startswith('newmtl'):
                    # save the previous material
                    if current_data_lines:
                        materials.append(cls._Parse(dataLines=current_data_lines, 
                                                    data_paths=all_paths, 
                                                    real_name=real_name, 
                                                    add_textures=add_textures,
                                                    is_debug_mode=debug_mode))
                        current_data_lines.clear()

                    # new name for the next new material
                    real_name = line.split(' ')[1]
                else:
                    current_data_lines.append(line)

            # save the last material
            if current_data_lines:
                materials.append(cls._Parse(dataLines=current_data_lines, 
                                            data_paths=all_paths, 
                                            real_name=real_name, 
                                            add_textures=add_textures,
                                            is_debug_mode=debug_mode))

            return tuple(materials)



__all__ = ['Material_MTL']