from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from static.data_types.color import Color
from .camera import Camera
from static.enums import LightType
from static.data_structure import (
    SingleLightData,
    FBO_Texture,
    ShadowLightData_PointLight,
    ShadowLightData_OtherLight
)

UNCHANGE = float('-inf')  # FLT_MIN


class Light:

    hasInitSSBO = False
    lightSSBO_ID = None
    ambientLightColor = None  # Set this appropriately
    ambientLightIntensity = None  # Set this appropriately
    lightsEnable = set()  # set to ensure uniqueness
    lightsCastingShadow = set()

    # shadow
    castShadow = False
    hasInitFrameBufferAndTexture = False
    allPointLightShadowFBOandTex = []  # static vector<FBO_Texture> allPointLightShadowFBOandTex;
    allOtherLightShadowFBOandTex = []  # static vector<FBO_Texture> allOtherLightShadowFBOandTex;
    frameBuffer_ID = None
    shadowMap_ID = None
    farPlane_pointLight = 100.0

    def __init__(self, enable=True, cast_shadow=False, light_type=None, light_color=None, intensity=1.0):
        self.thisLightData = SingleLightData()  # Define the structure appropriately
        if enable:
            Light.lightsEnable.add(self)
        self.thisLightData.lightType = light_type
        self.thisLightData.lightColor = light_color
        self.thisLightData.intensity = intensity
        if cast_shadow:
            self.set_cast_shadow(True)

    def __del__(self):
        Light.lightsEnable.remove(self)
        if self.is_cast_shadow():
            self.set_cast_shadow(False)

    def set_light_enable(self, set_val):
        if set_val and self not in Light.lightsEnable:
            Light.lightsEnable.append(self)
        elif not set_val and self in Light.lightsEnable:
            Light.lightsEnable.remove(self)

    @staticmethod
    def init_SSBO():
        if Light.hasInitSSBO:
            return
        # Equivalent GL functions, assuming you're using PyOpenGL
        Light.lightSSBO_ID = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, Light.lightSSBO_ID)
        # You might need to adjust the size calculation for glBufferData
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, Light.lightSSBO_ID)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        Light.hasInitSSBO = True

    def PreInitiFramebufferAndTexture():
        if not Light.hasInitFrameBufferAndTexture:
            # init point lights
            mapSize = Engine.GetShadowMapTextureSize()
            for i in range(Engine.GetMaxLightNumWithDepthMap()):
                fboTex = FBO_Texture()
                fboTex.FBO_ID = glGenFramebuffers(1)
                glBindFramebuffer(GL_FRAMEBUFFER, fboTex.FBO_ID)
                fboTex.shadowMapTexture_ID = glGenTextures(1)
                glBindTexture(GL_TEXTURE_CUBE_MAP, fboTex.shadowMapTexture_ID)
                for j in range(6):
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, 0, GL_DEPTH_COMPONENT, mapSize.x, mapSize.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
                    glTexParameterfv(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0)))
                glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, fboTex.shadowMapTexture_ID, 0)
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    print("ERROR when generating light shadow map framebuffer: not complete.")
                    return
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                Light.allPointLightShadowFBOandTex.append(fboTex)

            # init other lights
            for i in range(Engine.GetMaxLightNumWithDepthMap()):
                fboTex = FBO_Texture()
                fboTex.FBO_ID = glGenFramebuffers(1)
                glBindFramebuffer(GL_FRAMEBUFFER, fboTex.FBO_ID)
                fboTex.shadowMapTexture_ID = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, fboTex.shadowMapTexture_ID)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, mapSize.x, mapSize.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0)))
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboTex.shadowMapTexture_ID, 0)
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    print("ERROR when generating light shadow map framebuffer: not complete.")
                    return
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                Light.allOtherLightShadowFBOandTex.append(fboTex)

            print("Done generating all light shadow maps and textures.")
            Light.hasInitFrameBufferAndTexture = True

    def SetCastShadow(self, castShadow):
        if self.castShadow == castShadow:
            return

        if castShadow:
            if self.thisLightData.lightType == LightType.POINT_LIGHT:
                for fboTex in Light.allPointLightShadowFBOandTex:
                    if not fboTex.occupied:
                        self.frameBuffer_ID = fboTex.FBO_ID
                        self.shadowMap_ID = fboTex.shadowMapTexture_ID
                        fboTex.occupied = True
                        Light.lightsCastingShadow.append(self)
                        self.castShadow = True
                        return

                print("no free point light shadow map for using")
                return

            else:
                for fboTex in Light.allOtherLightShadowFBOandTex:
                    if not fboTex.occupied:
                        self.frameBuffer_ID = fboTex.FBO_ID
                        self.shadowMap_ID = fboTex.shadowMapTexture_ID
                        fboTex.occupied = True
                        Light.lightsCastingShadow.append(self)
                        self.castShadow = True
                        return

                print("no free other light shadow map for using")
                return

        else:
            Light.lightsCastingShadow.remove(self)
            if self.thisLightData.lightType == LightType.POINT_LIGHT:
                for fboTex in Light.allPointLightShadowFBOandTex:
                    if fboTex.FBO_ID == self.frameBuffer_ID:
                        fboTex.occupied = False
                        self.frameBuffer_ID = None
                        self.shadowMap_ID = None
                        return
            else:
                for fboTex in Light.allOtherLightShadowFBOandTex:
                    if fboTex.FBO_ID == self.frameBuffer_ID:
                        fboTex.occupied = False
                        self.frameBuffer_ID = None
                        self.shadowMap_ID = None
                        return
            glDeleteTextures(1, self.shadowMap_ID)

    def GetThisLightData(self):
        return self.thisLightData

    def SetLightType(self, light_type: LightType):
        if self.thisLightData.lightType == light_type:
            return
        if self.castShadow:
            self.SetCastShadow(False)
            self.thisLightData.lightType = light_type
            self.SetCastShadow(True)
        else:
            self.thisLightData.lightType = light_type

    def SetSpotLightCutOff(self, newCutOff=-1, newOuterCutOff=-1):
        if self.thisLightData.lightType != LightType.SPOT_LIGHT:
            print("This light is not a spot light. No need to set spot light cutoff.")
            return
        if newCutOff >= 0:
            self.thisLightData.cutOff = newCutOff
        if newOuterCutOff >= 0:
            self.thisLightData.outerCutOff = newOuterCutOff

    def SetDistanceFormula(self, quadratic=UNCHANGE, linear=UNCHANGE, constant=UNCHANGE):
        if self.thisLightData.lightType not in (LightType.POINT_LIGHT, LightType.SPOT_LIGHT):
            print("This light is not a point light or spot light. No need to set distance formula.")
            return
        if quadratic != self.UNCHANGE:
            self.thisLightData.quadratic = quadratic
        if linear != self.UNCHANGE:
            self.thisLightData.linear = linear
        if constant != self.UNCHANGE:
            self.thisLightData.constant = constant

    def SetLightColor(self, color: Color):
        self.thisLightData.lightColor = color

    def SetLightIntensity(self, intensity):
        if intensity < 0:
            print("Light intensity cannot be negative.")
            return
        self.thisLightData.intensity = intensity

    def SetPointLightShadowMapFarPlane(self, farPlane):
        if self.thisLightData.lightType != "POINT_LIGHT":
            print("This light is not a point light. No need to set point light shadow map far plane.")
            return
        elif farPlane < 0:
            print("Point light shadow map far plane cannot be negative.")
            return
        self.farPlane_pointLight = farPlane

    def isShadowLight(self):
        return self.castShadow

    @staticmethod
    def SetAmbientLight(color, intensity=UNCHANGE):
        global ambientLightColor, ambientLightIntensity  # Assuming these are global variables in your code
        ambientLightColor = color
        if intensity != Light.UNCHANGE:
            ambientLightIntensity = intensity

    def get_this_shadow_light_data_other_light(self):
        if not Light.castShadow or self.thisLightData.lightType == LightType.POINT_LIGHT:
            return ShadowLightData_OtherLight()

        shadowLightData = ShadowLightData_OtherLight()
        shadowLightData.frameBuffer_ID = self.frameBuffer_ID
        shadowLightData.shadowMap_ID = self.shadowMap_ID
        shadowLightData.lightVP = Mat4(1.0)
        shadowLightData.lightType = self.thisLightData.lightType

        if self.thisLightData.lightType == LightType.SPOT_LIGHT:
            shadowLightData.lightPos_OR_lightDir = self.thisLightData.position
        elif self.thisLightData.lightType == LightType.DIRECTIONAL_LIGHT:
            shadowLightData.lightPos_OR_lightDir = self.thisLightData.direction

        if not Camera.has_main_cam():
            return shadowLightData

        cam = Camera.get_main_cam()
        v = Mat4(1.0)
        p = Mat4(1.0)

        if self.thisLightData.lightType == LightType.DIRECTIONAL_LIGHT:
            scale = (Engine.get_shadow_map_texture_size().x / float(Engine.get_shadow_map_texture_size().y)) / 2
            p = glOrtho(-scale * Engine.get_ortho_distance(), scale * Engine.get_ortho_distance(), -scale *
                        Engine.get_ortho_distance(), scale * Engine.get_ortho_distance(), cam.nearPlane, cam.farPlane)
            gluLookAt(*(-10.0 * self.thisLightData.direction)._xyz, 0.0, 0.0, 0.0, *cam.get_up()._xyz)

        elif self.thisLightData.lightType == LightType.SPOT_LIGHT:
            p = gluPerspective(radians(self.thisLightData.outerCutOff * 2), Engine.get_shadow_map_texture_size().x / float(Engine.get_shadow_map_texture_size().y), cam.nearPlane, cam.farPlane)
            v = gluLookAt(*self.thisLightData.position, *(self.thisLightData.position + self.thisLightData.direction), *cam.get_up()._xyz)

        shadowLightData.lightVP = p * v
        return shadowLightData

    def GetThisShadowLightData_PointLight(self) -> ShadowLightData_PointLight:
        if not self.castShadow or self.thisLightData.lightType != LightType.POINT_LIGHT:
            return ShadowLightData_PointLight()

        shadowLightData = ShadowLightData_PointLight()
        shadowLightData.frameBuffer_ID = self.frameBuffer_ID
        shadowLightData.shadowMap_ID = self.shadowMap_ID
        shadowLightData.lightPos = self.thisLightData.position
        shadowLightData.farPlane = self.farPlane_pointLight
        shadowLightData.lightProj = gluPerspective(90.0, Engine.GetShadowMapTextureSize()[0] / float(Engine.GetShadowMapTextureSize()[1]), 0.0, self.farPlane_pointLight)
        position = self.thisLightData.position
        shadowLightData.lightViews = [
            gluLookAt(*position._xyz, position[0] + 1.0, position[1], position[2], 0.0, -1.0, 0.0),
            gluLookAt(*position._xyz, position[0] - 1.0, position[1], position[2], 0.0, -1.0, 0.0),
            gluLookAt(*position._xyz, position[0], position[1] + 1.0, position[2], 0.0, 0.0, 1.0),
            gluLookAt(*position._xyz, position[0], position[1] - 1.0, position[2], 0.0, 0.0, -1.0),
            gluLookAt(*position._xyz, position[0], position[1], position[2] + 1.0, 0.0, -1.0, 0.0),
            gluLookAt(*position._xyz, position[0], position[1], position[2] - 1.0, 0.0, -1.0, 0.0)
        ]
        return shadowLightData

    # Continue here ...
