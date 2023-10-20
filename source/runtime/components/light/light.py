from runtime.component import Component
from static import Color
from typing import Literal
import OpenGL.GL as gl

class Light(Component):
    '''The abstract class of all light components.'''

    _Shadow_Map_Dimension:Literal[2, 3] = 2
    '''Override this property to set the dimension of shadow map.'''
    _Light_Type_ID = None
    '''Override this property to set the light type ID, e.g. 0 for directional light, 1 for point light, 2 for spot light.'''

    def __init__(self,
                 gameObj,
                 enable=True,
                 lightCol:Color = Color.WHITE,
                 intensity:float = 1.0,
                 attenuation_constant:float = 1.0,
                 attenuation_linear:float = 0.09,
                 attenuation_quadratic:float = 0.032,
                 castShadow:bool = False, # if create shadow map
                 shadowMapSize:int = 1024,):
        super().__init__(gameObj, enable)
        self._lightCol = lightCol
        self._intensity = intensity
        self._attenuation_constant = attenuation_constant
        self._attenuation_linear = attenuation_linear
        self._attenuation_quadratic = attenuation_quadratic
        self._shadowMapSize = shadowMapSize
        self._castShadow = castShadow
        self._shadowMapID = None
        if castShadow:
            self._createShadowMap()

    def _createShadowMap(self):
        assert self._Shadow_Map_Dimension in (2, 3)
        if self._shadowMapID is not None:
            return # already created
        if self._Shadow_Map_Dimension == 2:
            # TODO
            pass
