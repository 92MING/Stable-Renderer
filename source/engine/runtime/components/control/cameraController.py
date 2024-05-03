import glm

from engine.static.enums import GLFW_Key, MouseButton
from ..camera.camera import Camera
from ...component import Component

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.runtime.gameObj import GameObject


class CameraController(Component):
    '''
    The default camera controlling component(by mouse).

    Control:
        * Left mouse button: move the camera
        * Right mouse button: rotate the camera
        * Mouse scroll: zoom in/out
        * R: reset the camera to default position and rotation
    '''

    RequireComponent = (Camera, )

    def __init__(self,
              gameObj: 'GameObject',
              enable=True,
              moveSpd = 0.005,
              zoomSpd = 0.2,
              horizontalRotSpd = 0.075,
              verticalRotSpd = 0.06,
              defaultPos: glm.vec3=None,
              defaultLookAt: glm.vec3=None,):
        super().__init__(gameObj, enable=enable)
        if defaultPos and not isinstance(defaultPos, glm.vec3) and isinstance(defaultPos, (list, tuple)):
            defaultPos = glm.vec3(defaultPos)
        if defaultLookAt and not isinstance(defaultLookAt, glm.vec3) and isinstance(defaultLookAt, (list, tuple)):
            defaultLookAt = glm.vec3(defaultLookAt)
        self.defaultPos = defaultPos
        self.defaultLookAt = defaultLookAt
        self.moveSpd = moveSpd
        self.zoomSpd = zoomSpd
        self.horizontalRotSpd = horizontalRotSpd
        self.verticalRotSpd = verticalRotSpd
        self._camera = None

    @property
    def camera(self) -> Camera:
        if self._camera is None:
            self._camera = self.gameObj.getComponent(Camera)
        return self._camera

    def reset(self):
        if self.defaultPos is not None:
            self.transform.position = self.defaultPos
        if self.defaultLookAt is not None:
            self.transform.lookAt(self.defaultLookAt)

    def start(self):
        self.reset()

    def update(self):
        inputManager = self.engine.InputManager
        if inputManager.GetKeyDown(GLFW_Key.R):
            self.reset()

        elif inputManager.HasMouseScrolled:
            self.transform.position = self.transform.position + self.transform.forward * inputManager.MouseScroll[1] * self.zoomSpd

        else:
            mouseDelta = inputManager.MouseDelta
            deltaX, deltaY = mouseDelta

            if inputManager.GetMouseBtn(MouseButton.LEFT):
                up = self.transform.up
                right = self.transform.right
                self.transform.position = self.transform.position + right * deltaX * self.moveSpd + up * deltaY * self.moveSpd

            elif inputManager.GetMouseBtn(MouseButton.RIGHT):
                self.transform.rotateLocalY(deltaX * self.horizontalRotSpd)
                self.transform.rotateLocalX(deltaY * self.verticalRotSpd)


__all__ = ['CameraController']