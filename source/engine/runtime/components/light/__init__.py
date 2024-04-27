'''All light components, including: DirectionalLight, PointLight, SpotLight, etc.'''
from .light import *

from .directional_light import *

from .point_light import *

from .spot_light import *

from engine.static.shader import Shader as _Shader

for subLightType in Light.AllLightSubTypes():
    # edit max light number constant. E.g. #define MAX_POINTLIGHT 256
    name = subLightType.__qualname__.split('.')[-1].upper()
    _Shader.SetShaderConstant(f'MAX_{name}', subLightType.Max_Num())