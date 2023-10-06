from typing import List, Tuple, Dict, Union, Callable
import numpy as np
from .color import ConstColor
from .data_types.vector import Vector
from .enums import LightType


class Vertex:
    def __init__(self, position, uv, normal):
        self.position = position
        self.uv = uv
        self.normal = normal


class KeyCallbackData:
    def __init__(self, key: int, action: int):
        self.key = key
        self.action = action


class MouseCallbackData:
    def __init__(self, button: int, action: int):
        self.button = button
        self.action = action


class ScrollCallbackData:
    def __init__(self, x_offset: float, y_offset: float):
        self.x_offset = x_offset
        self.y_offset = y_offset


class CursorPositionCallbackData:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class CameraVPData:
    def __init__(self, camera_position, view_matrix, projection_matrix):
        self.camera_position = camera_position
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix


# Placeholder for undefined types

class GameShader:
    pass


class GameMesh:
    pass


class GameTexture:
    pass


class SingleLightData:
    def __init__(self):
        self.lightType = LightType.POINT_LIGHT
        self.direction = Vector(-1.0, -1.0, -1.0)
        self.position = Vector(0.0, 0.0, 0.0)
        self.cutOff = 30.0
        self.outerCutOff = 45.0
        self.constant = 0.0
        self.linear = 0.0
        self.quadratic = 1.0
        self.lightColor = ConstColor.WHITE
        self.intensity = 1.0


class LightData_SSBO:
    def __init__(self):
        self.ambientPower = 1.0
        self.ambientLight = ConstColor.BLACK
        self.singleLightDataLength = 0
        self.allSingleLightData: List[SingleLightData] = []


class ShadowLightData_PointLight:
    def __init__(self):
        self.lightProj = np.eye(4)  # 4x4 identity matrix as placeholder
        self.lightViews = [np.eye(4) for _ in range(6)]
        self.frameBuffer_ID = 0
        self.shadowMap_ID = 0
        self.farPlane = 0.0
        self.lightPos = np.array([0.0, 0.0, 0.0])


class ShadowLightData_OtherLight:
    def __init__(self):
        self.lightVP = np.eye(4)
        self.frameBuffer_ID = 0
        self.shadowMap_ID = 0
        self.lightType = ""  # Placeholder
        self.lightPos_OR_lightDir = Vector(0.0, 0.0, 0.0)


class FBO_Texture:
    def __init__(self):
        self.FBO_ID = 0
        self.shadowMapTexture_ID = 0
        self.occupied = False


class CurrentTextureData:
    def __init__(self):
        self.slot = 0
        self.textureID = 0
        self.offset = Vector(0.0, 0.0)
        self.uniformName = ""


class CurrentMaterialVariableData:
    def __init__(self):
        self.uniformName = ""
        self.variableType = ""  # Placeholder
        self.value = None


class CurrentMaterialData:
    def __init__(self):
        self.shader = None  # Placeholder
        self.allTextureData: List[CurrentTextureData] = []
        self.allVariableData: List[CurrentMaterialVariableData] = []
        self.emissionColor = ConstColor.BLACK
        self.emissionIntensity = 1.0
        self.receiveShadow = True


class CurrentMeshData:
    def __init__(self):
        self.mesh = None  # Placeholder
        self.modelMatrix = np.eye(4)


class RenderFrameData:
    def __init__(self):
        self.opaqueRenderFuncs: List[Tuple[Callable, GameShader]] = []
        self.transparentRenderFuncs: Dict[int, Tuple[Callable, GameShader]] = {}
        self.uiRenderFuncs: List[Tuple[Callable, GameShader]] = []
        self.allSingleLights: List[SingleLightData] = []
        self.ambientLight = ConstColor.BLACK
        self.ambientPower = 1.0
        self.allShadowLights_DirectionalLight: List[ShadowLightData_OtherLight] = []
        self.allShadowLights_SpotLight: List[ShadowLightData_OtherLight] = []
        self.allShadowLights_PointLight: List[ShadowLightData_PointLight] = []
        self.allMeshesCastingShadow: List[Tuple[CurrentMeshData, GameTexture]] = []
        self.cameraVPData = None  # Placeholder
        self.cameraBackgroundColor = ConstColor.BLACK
