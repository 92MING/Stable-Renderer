from .light import Light
from typing import Literal

class DirectionalLight(Light):

    _Shadow_Map_Dimension: Literal[2, 3] = 2
    _Light_Type_ID = 0
    '''The light type ID of directional light is 0. This value will be used in shader'''