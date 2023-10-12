from runtime.components.transform import *
from static.color import Color
from static.enums import ProjectionType
from typing import Set
import glm

class Camera(Component):

    # region class properties / methods
    _Main_Camera = None # current main camera
    _All_Cameras:Set['Camera'] = set()
    _Has_Init_UBO = False
    _Camera_UBO_ID = None

    RequireComponent = (Transform, )

    @classmethod
    def ActiveCameras(cls):
        return [cam for cam in cls._All_Cameras if cam.enable]
    @classmethod
    def MainCamera(cls):
        return cls._Main_Camera
    # endregion

    def __init__(self,
                 gameObj,
                 enable=True,
                 fov=45.0,
                 near_plane=0.1,
                 far_plane=100.0,
                 ortho_size=1.0,
                 bgColor=Color.CLEAR,
                 projection_type=ProjectionType.PERSPECTIVE):
        super().__init__(gameObj, enable)
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.ortho_size = ortho_size
        self.bgColor = bgColor
        self.projection_type = projection_type

    def awake(self):
        Camera._All_Cameras.add(self)
    def onDestroy(self):
        Camera._All_Cameras.remove(self)
    def onDisable(self):
        if self.isMainCamera:
            Camera._Main_Camera = None
            for cam in Camera._All_Cameras: # try to set another camera as main camera
                if cam is not self:
                    if cam.set_as_main_camera():
                        break

    @property
    def isMainCamera(self):
        return Camera._Main_Camera == self
    def set_as_main_camera(self)->bool:
        '''
        Set this camera as the main camera. Return True if success, otherwise return False.
        :return: bool
        '''
        if not self.enable:
            return False
        Camera._Main_Camera = self
        return True

    @property
    def _pos(self):
        return self.gameObj.transform.globalPos
    @property
    def _forward(self):
        return self.gameObj.transform.forward
    @property
    def _up(self):
        return self.gameObj.transform.up
    @property
    def viewMatrix(self):
        return glm.lookAt(self._pos, self._pos + self._forward, self._up)
    @property
    def projectionMatrix(self):
        if self.projection_type == ProjectionType.PERSPECTIVE:
            return glm.perspective(
                self.fov,
                self.engine.WindowManager.AspectRatio,
                self.near_plane,
                self.far_plane,
            )
        elif self.projection_type == ProjectionType.ORTHOGRAPHIC:
            screen_scale = self.engine.WindowManager.AspectRatio * self.ortho_size / 2
            return glm.ortho(
                -screen_scale,
                screen_scale,
                -self.ortho_size / 2,
                self.ortho_size / 2,
                self.near_plane,
                self.far_plane,
            )

    def lateUpdate(self):
        if self.isMainCamera:
            if self.engine.bgColor != self.bgColor:
                self.engine.bgColor = self.bgColor
            viewMatrix, projectionMatrix = self.viewMatrix, self.projectionMatrix
            if self.engine.RenderManager.UBO_ViewMatrix != viewMatrix:
                self.engine.RenderManager.UpdateUBO_ViewMatrix(viewMatrix)
            if self.engine.RenderManager.UBO_ProjMatrix != projectionMatrix:
                self.engine.RenderManager.UpdateUBO_ProjMatrix(projectionMatrix)
