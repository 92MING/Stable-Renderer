import OpenGL.GLUT as glut
import sys
import OpenGL.GL as gl
import OpenGL.GLU as glu
from static.color import *
from static.enums import *
from utils.data_struct import Event
from static.scene import *


class DelayEvent(Event):
    '''
    When invoke delay event, it will not be executed immediately, but params will be save and execute later by calling "release".
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_param = None
    def invoke(self, *args):
        self._check_invoke_params(*args)
        self._last_param = args
    def release(self):
        if self._last_param is not None:
            super().invoke(*self._last_param)
        self._last_param = None

class GlutWindow(object):

    def __init__(self,
                 winTitle='Stable Renderer',
                 winSize=(800, 480),
                 bgColor=Color.CLEAR,
                 depthTest=True,
                 depthFunc=DepthFunc.LESS, ):

        self._winTitle = winTitle
        self._winSize = winSize
        glut.glutInit(sys.argv)
        self.setBgColor(bgColor)
        self.setDepthTest(depthTest)
        self.setDepthFunc(depthFunc)

        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH | glut.GLUT_STENCIL)
        glut.glutInitWindowSize(*self.winSize)
        self.window = glut.glutCreateWindow(self.winTitle.encode())

        self._curMousePos = (0, 0)
        self._curMouseKey = None
        self._curMouseAction = None
        self._curKeyMod = None
        self._curKeys = {} # key: state
        self.windowResizeEvent = DelayEvent(int, int)

        glut.glutDisplayFunc(self.display)
        glut.glutReshapeFunc(self.resize)
        glut.glutKeyboardFunc(self.on_keyboard)
        glut.glutSpecialFunc(self.on_special_key)
        glut.glutMouseFunc(self.on_mouse)
        glut.glutMotionFunc(self.on_mousemove)
        glut.glutPassiveMotionFunc(self.on_mousemove)

    # region properties
    @property
    def winTitle(self):
        return self._winTitle

    @property
    def winSize(self):
        return self._winSize

    @property
    def depthFunc(self):
        return self._depthFunc

    @depthFunc.setter
    def depthFunc(self, value):
        self._depthFunc = value
        gl.glDepthFunc(self._depthFunc.value)

    def setDepthFunc(self, depthFunc: DepthFunc):
        self.depthFunc = depthFunc

    @property
    def depthTest(self):
        return self._depthTest

    @depthTest.setter
    def depthTest(self, value):
        self._depthTest = value
        if self._depthTest:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def setDepthTest(self, enable: bool):
        self.depthTest = enable

    @property
    def bgColor(self):
        return self._bgColor

    @bgColor.setter
    def bgColor(self, value):
        self._bgColor = value
        gl.glClearColor(self._bgColor.r, self._bgColor.g, self._bgColor.b, self._bgColor.a)

    def setBgColor(self, color: Color):
        self.bgColor = color

    # endregion

    def run(self):
        glut.glutMainLoop()
    def stop(self):
        glut.glutLeaveMainLoop()

    # region events
    def _display_func(self):

        self.windowResizeEvent.release()

        # TODO: update

        # TODO: late update

        glut.glutSwapBuffers()

    def _idle_func(self):

        # TODO: render

        glut.glutPostRedisplay()

    def _on_window_resize(self, width, height):
        self._winSize = (width, height)
        self.windowResizeEvent.invoke(width, height)
    def _on_key_down(self, key, x, y):
        if glut.glutGetModifiers() != 0:
            self._curKeyMod = KeyMod.GetEnum(glut.glutGetModifiers())
        else:
            self._curKey = Key.GetEnum(key)
    def
    def _on_mouse(self, button, state, x, y):
        self._curMousePos = (x, y)
        self._curMouseKey = MouseButton.GetEnum(button)
        self._curMouseAction = MouseAction.GetEnum(state)
    def _on_mousemove(self, x, y):
        self._curMousePos = (x, y)
    # endregion


if __name__ == "__main__":
    win = GlutWindow()
    win.run()