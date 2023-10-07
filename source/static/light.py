import numpy as np
from typing import List
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from .engine import Engine
from .data_types.vector import Vector
from .color import Color
from .camera import Camera
from .enums import LightType
from .data_structure import (
    SingleLightData,
    FBO_Texture,
    ShadowLightDataPointLight,
    ShadowLightDataOtherLight,
    LightData_SSBO
)
from .data_types.matrix import Matrix


class Light:
    UNCHANGE = float('-inf')  # FLT_MIN

    has_init_SSBO = False
    light_SSBO_id = None
    ambient_light_color = None  # Set this appropriately
    ambient_light_intensity = None  # Set this appropriately
    lights_enable = set()  # set to ensure uniqueness
    lights_casting_shadow = set()

    # shadow
    cast_shadow = False
    has_init_frame_buffer_and_texture = False
    all_point_light_shadow_FBO_and_tex: List[FBO_Texture] = []  # static vector<FBO_Texture> allPointLightShadowFBOandTex;
    all_other_light_shadow_FBO_and_tex: List[FBO_Texture] = []  # static vector<FBO_Texture> allOtherLightShadowFBOandTex;
    frame_buffer_id = None
    shadow_map_id = None
    far_plane_point_light = 100.0

    def __init__(self, enable=True, cast_shadow=False, light_type=None, light_color=None, intensity=1.0):
        self.this_light_data = SingleLightData()  # Define the structure appropriately
        if enable:
            Light.lights_enable.add(self)
        self.this_light_data.light_type = light_type
        self.this_light_data.lightColor = light_color
        self.this_light_data.intensity = intensity
        if cast_shadow:
            self.set_cast_shadow(True)

    def __del__(self):
        Light.lights_enable.remove(self)
        if self.is_cast_shadow():
            self.set_cast_shadow(False)

    def set_light_enable(self, set_val):
        if set_val and self not in Light.lights_enable:
            Light.lights_enable.append(self)
        elif not set_val and self in Light.lights_enable:
            Light.lights_enable.remove(self)

    @staticmethod
    def init_SSBO():
        if Light.has_init_SSBO:
            return
        # Equivalent GL functions, assuming you're using PyOpenGL
        Light.light_SSBO_id = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, Light.light_SSBO_id)
        # You might need to adjust the size calculation for glBufferData
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, Light.light_SSBO_id)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        Light.has_init_SSBO = True

    def pre_init_framebuffer_and_texture():
        if not Light.has_init_frame_buffer_and_texture:
            # init point lights
            max_size = Engine.get_shadow_map_texture_size()
            for i in range(Engine.get_max_light_num_with_depth_map()):
                FBO_tex = FBO_Texture()
                FBO_tex.FBO_id = glGenFramebuffers(1)
                glBindFramebuffer(GL_FRAMEBUFFER, FBO_tex.FBO_id)
                FBO_tex.shadow_map_texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_CUBE_MAP, FBO_tex.shadow_map_texture_id)
                for j in range(6):
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, 0, GL_DEPTH_COMPONENT, max_size.x, max_size.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
                    glTexParameterfv(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BORDER_COLOR, np.array([1.0, 1.0, 1.0, 1.0]))  # TODO: 不确定最后一个参数是否正确
                glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, FBO_tex.shadow_map_texture_id, 0)
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    print("ERROR when generating light shadow map framebuffer: not complete.")
                    return
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                Light.all_point_light_shadow_FBO_and_tex.append(FBO_tex)

            # init other lights
            for i in range(Engine.get_max_light_num_with_depth_map()):
                FBO_tex = FBO_Texture()
                FBO_tex.FBO_id = glGenFramebuffers(1)
                glBindFramebuffer(GL_FRAMEBUFFER, FBO_tex.FBO_id)
                FBO_tex.shadow_map_texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, FBO_tex.shadow_map_texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, max_size.x, max_size.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.array([1.0, 1.0, 1.0, 1.0]))
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, FBO_tex.shadow_map_texture_id, 0)
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    print("ERROR when generating light shadow map framebuffer: not complete.")
                    return
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                Light.all_other_light_shadow_FBO_and_tex.append(FBO_tex)

            print("Done generating all light shadow maps and textures.")
            Light.has_init_frame_buffer_and_texture = True

    def set_cast_shadow(self, castShadow):
        if self.cast_shadow == castShadow:
            return

        if castShadow:
            if self.this_light_data.light_type == LightType.POINT:
                for FBO_tex in Light.all_point_light_shadow_FBO_and_tex:
                    if not FBO_tex.occupied:
                        self.frame_buffer_id = FBO_tex.FBO_id
                        self.shadow_map_id = FBO_tex.shadow_map_texture_id
                        FBO_tex.occupied = True
                        Light.lights_casting_shadow.append(self)
                        self.cast_shadow = True
                        return

                print("no free point light shadow map for using")
                return

            else:
                for FBO_tex in Light.all_other_light_shadow_FBO_and_tex:
                    if not FBO_tex.occupied:
                        self.frame_buffer_id = FBO_tex.FBO_id
                        self.shadow_map_id = FBO_tex.shadow_map_texture_id
                        FBO_tex.occupied = True
                        Light.lights_casting_shadow.append(self)
                        self.cast_shadow = True
                        return

                print("no free other light shadow map for using")
                return

        else:
            Light.lights_casting_shadow.remove(self)
            if self.this_light_data.light_type == LightType.POINT:
                for FBO_tex in Light.all_point_light_shadow_FBO_and_tex:
                    if FBO_tex.FBO_id == self.frame_buffer_id:
                        FBO_tex.occupied = False
                        self.frame_buffer_id = None
                        self.shadow_map_id = None
                        return
            else:
                for FBO_tex in Light.all_other_light_shadow_FBO_and_tex:
                    if FBO_tex.FBO_id == self.frame_buffer_id:
                        FBO_tex.occupied = False
                        self.frame_buffer_id = None
                        self.shadow_map_id = None
                        return
            glDeleteTextures(1, self.shadow_map_id)

    def set_light_type(self, light_type: LightType):
        if self.this_light_data.light_type == light_type:
            return
        if self.cast_shadow:
            self.set_cast_shadow(False)
            self.this_light_data.light_type = light_type
            self.set_cast_shadow(True)
        else:
            self.this_light_data.light_type = light_type

    def set_spot_light_cutoff(self, new_cutoff=-1, new_outer_cutoff=-1):
        if self.this_light_data.light_type != LightType.SPOT:
            print("This light is not a spot light. No need to set spot light cutoff.")
            return
        if new_cutoff >= 0:
            self.this_light_data.cutOff = new_cutoff
        if new_outer_cutoff >= 0:
            self.this_light_data.outer_cutoff = new_outer_cutoff

    def set_distance_formula(self, quadratic=UNCHANGE, linear=UNCHANGE, constant=UNCHANGE):
        if self.this_light_data.light_type not in (LightType.POINT, LightType.SPOT):
            print("This light is not a point light or spot light. No need to set distance formula.")
            return
        if quadratic != self.UNCHANGE:
            self.this_light_data.quadratic = quadratic
        if linear != self.UNCHANGE:
            self.this_light_data.linear = linear
        if constant != self.UNCHANGE:
            self.this_light_data.constant = constant

    def set_light_color(self, color: Color):
        self.this_light_data.lightColor = color

    def set_light_intensity(self, intensity):
        if intensity < 0:
            print("Light intensity cannot be negative.")
            return
        self.this_light_data.intensity = intensity

    def set_point_light_shadow_map_far_plane(self, farPlane):
        if self.this_light_data.light_type != "POINT_LIGHT":
            print("This light is not a point light. No need to set point light shadow map far plane.")
            return
        elif farPlane < 0:
            print("Point light shadow map far plane cannot be negative.")
            return
        self.far_plane_point_light = farPlane

    def isShadowLight(self):
        return self.cast_shadow

    @ staticmethod
    def set_ambient_light(color, intensity=UNCHANGE):
        global ambient_light_color, ambient_light_intensity  # Assuming these are global variables in your code
        ambient_light_color = color
        if intensity != Light.UNCHANGE:
            ambient_light_intensity = intensity

    def get_this_shadow_light_data_other_light(self):
        if not Light.cast_shadow or self.this_light_data.light_type == LightType.POINT:
            return ShadowLightDataOtherLight()

        shadow_light_data = ShadowLightDataOtherLight()
        shadow_light_data.frame_buffer_id = self.frame_buffer_id
        shadow_light_data.shadow_map_id = self.shadow_map_id
        shadow_light_data.light_VP = Matrix.Identity(4)
        shadow_light_data.light_type = self.this_light_data.light_type

        if self.this_light_data.light_type == LightType.SPOT:
            shadow_light_data.light_pos_or_light_dir = self.this_light_data.position
        elif self.this_light_data.light_type == LightType.DIRECTIONAL:
            shadow_light_data.light_pos_or_light_dir = self.this_light_data.direction

        if not Camera.has_main_cam():
            return shadow_light_data

        cam: Camera = Camera.get_main_cam()
        v = Matrix.Identity(4)
        p = Matrix.Identity(4)

        if self.this_light_data.light_type == LightType.DIRECTIONAL:
            scale = (Engine.get_shadow_map_texture_size().x / float(Engine.get_shadow_map_texture_size().y)) / 2
            p = Matrix.Orthographic(-scale * Engine.get_ortho_distance(), scale * Engine.get_ortho_distance(), -scale *
                                    Engine.get_ortho_distance(), scale * Engine.get_ortho_distance(), cam.nearPlane, cam.farPlane)
            Matrix.LookAt(-10.0 * self.this_light_data.direction, Vector(0.0, 0.0, 0.0), cam.get_up())

        elif self.this_light_data.light_type == LightType.SPOT:
            p = Matrix.Perspective(
                np.radians(self.this_light_data.outer_cutoff * 2),
                Engine.get_shadow_map_texture_size().x / float(Engine.get_shadow_map_texture_size().y),
                cam.near_plane,
                cam.far_plane
            )
            v = Matrix.LookAt(
                self.this_light_data.position,
                self.this_light_data.position + self.this_light_data.direction,
                cam.get_up()
            )

        shadow_light_data.light_VP = p * v
        return shadow_light_data

    def get_this_shadow_light_data_point_light(self) -> ShadowLightDataPointLight:
        if not self.cast_shadow or self.this_light_data.light_type != LightType.POINT:
            return ShadowLightDataPointLight()

        shadow_light_data = ShadowLightDataPointLight()
        shadow_light_data.frame_buffer_id = self.frame_buffer_id
        shadow_light_data.shadow_map_id = self.shadow_map_id
        shadow_light_data.light_pos = self.this_light_data.position
        shadow_light_data.far_plane = self.far_plane_point_light
        shadow_light_data.light_proj = Matrix.Perspective(
            90.0,
            Engine.get_shadow_map_texture_size()[0] / float(Engine.get_shadow_map_texture_size()[1]),
            0.0,
            self.far_plane_point_light
        )
        position = self.this_light_data.position
        shadow_light_data.light_views = [
            Matrix.LookAt(position, position + Vector(1.0, 0.0, 0.0), Vector(0.0, -1.0, 0.0)),
            Matrix.LookAt(position, position + Vector(-1.0, 0.0, 0.0), Vector(0.0, -1.0, 0.0)),
            Matrix.LookAt(position, position + Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)),
            Matrix.LookAt(position, position + Vector(0.0, -1.0, 0.0), Vector(0.0, 0.0, -1.0)),
            Matrix.LookAt(position, position + Vector(0.0, 0.0, 1.0), Vector(0.0, -1.0, 0.0)),
            Matrix.LookAt(position, position + Vector(0.0, 0.0, -1.0), Vector(0.0, -1.0, 0.0))
        ]
        return shadow_light_data

    # these methods may be unnecessary
    @ classmethod
    def get_light_casting_shadow(cls):
        return cls.lights_casting_shadow

    @ classmethod
    def get_lights_enable(cls):
        return cls.lights_enable

    @ classmethod
    def current_enable_light_num(cls):
        return len(cls.lights_enable)

    @ classmethod
    def current_light_using_depth_map_num(cls):
        return len(cls.lights_casting_shadow)

    @ classmethod
    def get_ambient_light_power(cls):
        return cls.ambient_light_intensity

    @ classmethod
    def get_ambient_light_color(cls):
        return cls.ambient_light_color

    @ classmethod
    def get_light_SSBO_id(cls):
        if not cls.has_init_SSBO:
            cls.init_SSBO()  # You'll need to define InitSSBO method in Python
        return cls.light_SSBO_id

    @ classmethod
    def set_all_SSBO_data(cls, all_lights, ambient_light, ambient_power):
        light_data_SSBO = LightData_SSBO()  # This should be a dict or another class
        light_data_SSBO.all_single_light_data = all_lights
        light_data_SSBO.single_light_data_length = len(all_lights)
        light_data_SSBO.ambient_light = ambient_light
        light_data_SSBO.ambient_power = ambient_power

        if not cls.has_init_SSBO:
            cls.init_SSBO()

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cls.light_SSBO_id)

        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        sys.getsizeof(float) + sys.getsizeof(Color) + sys.getsizeof(int) + sys.getsizeof(all_lights),
                        light_data_SSBO
                        )

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    @ classmethod
    def delete_light_SSBO(cls):
        if cls.has_init_SSBO:
            glDeleteBuffers(1, cls.light_SSBO_id)
            cls.has_init_SSBO = False
