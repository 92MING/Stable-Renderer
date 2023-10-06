import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from .color import Color
from .data_types.vector import Vector
from .enums import ProjectionType
# from utils.global_utils import GetOrAddGlobalValue

# _ALL_CAMERAS: dict = GetOrAddGlobalValue('_ALL_CAMERAS', dict())


class CameraVPData:
    def __init__(self, camera_position, view_matrix, projection_matrix):
        self.camera_position = camera_position
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix


class Camera:
    main_cam = None
    all_cams = set()
    active_cams = []
    has_init_UBO = False
    cam_UBO = None

    def __init__(self):
        self.all_cams.add(self)
        self.fov = 90.0
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.ortho_size = 1.0
        self.background_color = Color.Black
        self.projection_type = "perspective"

    def __del__(self):
        self.all_cams.remove(self)
        self.change_camera_active_state(False)

    def change_camera_active_state(self, active):
        if active and not self.is_active_cam():
            self.active_cams.append(self)
            if len(self.active_cams) == 1:
                self.main_cam = self
        elif self.is_active_cam():
            self.active_cams.remove(self)
            if self.main_cam == self:
                if self.active_cams:
                    self.main_cam = self.active_cams[0]
                else:
                    self.main_cam = None

    def is_main_cam(self):
        return self.main_cam is self

    def is_active_cam(self):
        return self in self.active_cams

    def set_as_main_cam(self):
        if not self.is_active_cam():
            return
        self.main_cam = self

    def get_pos(self):
        return Vector(0.0, 0.0, 0.0)

    def get_forward(self):
        return Vector(1.0, 0.0, 0.0)

    def get_up(self):
        return Vector(0.0, 1.0, 0.0)

    def get_scene_width_and_height_near_plane(self, aspect_ratio):
        width = 2 * np.tan(self.fov / 2) * self.near_plane
        height = width / aspect_ratio
        return width, height

    def get_view_matrix(self):
        return gluLookAt(self.get_pos(), self.get_pos() + self.get_forward(), self.get_up())

    def get_projection_matrix(self, window_width, window_height, ortho_distance):
        if self.projection_type == ProjectionType.PERSPECTIVE:
            return gluPerspective(np.radians(self.fov), window_width / window_height, self.near_plane, self.far_plane)
        elif self.projection_type == ProjectionType.ORTHOGRAPHIC:
            screenScale = window_width / window_height * self.ortho_size / 2
            return glOrtho(-screenScale * ortho_distance, screenScale * ortho_distance, -float(ortho_distance), float(ortho_distance), self.near_plane, self.far_plane)

    @staticmethod
    def init_cam_UBO():
        if Camera.has_init_UBO:
            return
        Camera.cam_UBO = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, Camera.cam_UBO)
        glBufferData(GL_UNIFORM_BUFFER, sys.getsizeof(CameraVPData), None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, Camera.cam_UBO)
        Camera.has_init_UBO = True

    @staticmethod
    def update_UBO(data):
        if not Camera.has_init_UBO:
            Camera.init_cam_UBO()
        glBindBuffer(GL_UNIFORM_BUFFER, Camera.cam_UBO)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CameraVPData), data)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    @staticmethod
    def delete_cam_UBO():
        if not Camera.has_init_UBO:
            return
        glDeleteBuffers(1, Camera.cam_UBO)
        Camera.has_init_UBO = False

    @staticmethod
    def get_cam_UBO_id() -> GLuint:
        if not Camera.has_init_UBO:
            Camera.init_cam_UBO()
        return Camera.cam_UBO

    @staticmethod
    def get_main_cam():
        return Camera.main_cam

    @staticmethod
    def has_main_cam():
        return Camera.main_cam is not None


class CameraAttribute(GameAttribute, Camera):
    def __init__(self, gameObject, enable):
        GameAttribute.__init__(self, gameObject, enable)
        Camera.__init__(self)

    def get_pos(self):
        return self.gameObject.transform().GetWorldPos()

    def get_forward(self):
        return self.gameObject.transform().forward()

    def get_up(self):
        return self.gameObject.transform().up()

    def on_enable(self):
        self.change_camera_active_state(True)

    def on_disable(self):
        self.change_camera_active_state(False)

    def late_update(self):
        if self.is_main_cam():
            Engine.SetCameraDataToRenderFrameData(CameraVPData(self.get_pos(), self.get_view_matrix(), self.get_projection_matrix()), self.background_color)
