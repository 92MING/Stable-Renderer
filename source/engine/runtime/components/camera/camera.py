from typing import Set, TYPE_CHECKING
import glm

from ...component import Component
from ..transform import Transform
from engine.static.color import Color
from engine.static.enums import ProjectionType

if TYPE_CHECKING:
    from engine.managers import RuntimeManager


class Camera(Component):

    # region class properties / methods
    _Main_Camera = None # current main camera
    _All_Cameras:Set['Camera'] = set()

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
    def onEnable(self):
        if Camera._Main_Camera is None:
            self.set_as_main_camera()

    @property
    def isMainCamera(self):
        return Camera._Main_Camera == self
    def set_as_main_camera(self)->bool:
        '''
        Set this camera as the main camera. Return True if success, otherwise return False.
        :return: bool - whether this camera is set as the main camera
        '''
        if not self.enable:
            return False
        Camera._Main_Camera = self
        return True

    @property
    def viewMatrix(self):
        pos = self.transform.position
        forward = self.transform.forward
        up = self.transform.up
        return glm.lookAt(pos, pos + forward, up)
    @property
    def projectionMatrix(self):
        if self.projection_type == ProjectionType.PERSPECTIVE:
            fov = glm.radians(self.fov)
            return glm.perspective(
                fov,
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
            runtimeManager: 'RuntimeManager' = self.engine.RuntimeManager
            
            if self.engine.WindowManager.BgColor != self.bgColor:
                self.engine.WindowManager.BgColor = self.bgColor
                
            worldPos, worldForward = self.transform.position, self.transform.forward
            viewMatrix, projectionMatrix = self.viewMatrix, self.projectionMatrix
            
            if self.engine.RuntimeManager.UpdateUBO_ViewMatrix != viewMatrix:
                self.engine.RuntimeManager.UpdateUBO_ViewMatrix(viewMatrix)
            
            if self.engine.RuntimeManager.UpdateUBO_ProjMatrix != projectionMatrix:
                self.engine.RuntimeManager.UpdateUBO_ProjMatrix(projectionMatrix)
            
            runtimeManager.UpdateUBO_CamInfo(
                worldPos,
                worldForward,
                self.near_plane,
                self.far_plane,
                self.fov,
            )


__all__ = ['Camera']