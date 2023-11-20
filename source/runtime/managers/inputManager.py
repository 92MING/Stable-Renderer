from .manager import Manager
from static.enums import *
import glfw
from typing import Tuple

class InputManager(Manager):
    def __init__(self, window):
        super().__init__()
        self._mousePos = (0, 0)  # (x, y)
        self._mouseDelta = (0, 0)  # (dx, dy)
        self._mouseScroll = (0, 0)  # (dx, dy)
        self._mouseKeys = {}  # key: (action, modifier)
        self._keyboardKeys = {}  # key: (action, modifier)
        self._glfw_init(window)

    # region callbacks
    def _cursorCallback(self, window, x, y):
        # for glfw callback
        self._mouseDelta = (x - self._mousePos[0], y - self._mousePos[1])
        self._mousePos = (x, y)
    def _scrollCallback(self, window, x, y):
        # for glfw callback
        self._mouseScroll = (x, y)
    def _mouseButtonCallback(self, window, btn, action, mod):
        # for glfw callback
        btn = MouseButton.GetEnum(btn)
        action = InputAction.GetEnum(action)
        mod = InputModifier.GetEnum(mod)
        if btn not in self._mouseKeys:
            self._mouseKeys[btn] = [None, None]
        current = self._mouseKeys[btn]
        if action == InputAction.PRESS:
            if current[0] != InputAction.HOLD:
                current = [action, mod]
        elif action == InputAction.RELEASE:
            current = [action, mod]
        self._mouseKeys[btn] = current
    def _keyCallback(self, window, key, scancode, action, mod):
        # for glfw callback
        key = Key.GetEnum(key)
        if key not in self._keyboardKeys:
            self._keyboardKeys[key] = [None, None]  # (action, modifier)
        action = InputAction.GetEnum(action)
        mod = InputModifier.GetEnum(mod)
        current = self._keyboardKeys[key]
        if action == InputAction.PRESS:
            if current[0] != InputAction.HOLD:
                current = [action, mod]
        elif action == InputAction.RELEASE:
            current = [action, mod]
        self._keyboardKeys[key] = current
    # endregion

    def _glfw_init(self, window):
        glfw.set_cursor_pos_callback(window, self._cursorCallback)
        glfw.set_scroll_callback(window, self._scrollCallback)
        glfw.set_mouse_button_callback(window, self._mouseButtonCallback)
        glfw.set_key_callback(window, self._keyCallback)

    def _onFrameBegin(self):
        glfw.poll_events()  # update input info
    def _onFrameEnd(self):
        self._mouseDelta = (0, 0)
        self._mouseScroll = (0, 0)
        for key in self._mouseKeys.copy():
            cur = self._mouseKeys[key]
            if cur[0] == InputAction.RELEASE:
                self._mouseKeys.pop(key)
            elif cur[0] == InputAction.PRESS:
                self._mouseKeys[key][0] = InputAction.HOLD
        for key in self._keyboardKeys.copy():
            cur = self._keyboardKeys[key]
            if cur[0] == InputAction.RELEASE:
                self._keyboardKeys.pop(key)
            elif cur[0] == InputAction.PRESS:
                self._keyboardKeys[key][0] = InputAction.HOLD

    @property
    def MousePos(self)->Tuple[int, int]:
        '''Return mouse position.'''
        return tuple(self._mousePos)
    @property
    def MouseDelta(self)->Tuple[int, int]:
        '''Return mouse delta since last frame. e.g. (1, 0) means move right 1 unit.'''
        return tuple(self._mouseDelta)
    @property
    def MouseScroll(self)->Tuple[int, int]:
        '''Return mouse scroll since last frame. e.g. (0, 1) means scroll up 1 unit.'''
        return tuple(self._mouseScroll)
    @property
    def HasMouseMoved(self)->bool:
        '''Check if mouse has moved since last frame.'''
        return self._mouseDelta != (0, 0)
    @property
    def HasMouseScrolled(self)->bool:
        '''Check if mouse has scrolled since last frame.'''
        return self._mouseScroll != (0, 0)

    # region checking
    def GetMouseBtn(self, btn: MouseButton, mod: InputModifier = None):
        '''Check if mouse btn is pressed. If mod is not None, any modifier will be accepted.'''
        if btn not in self._mouseKeys:
            return False
        else:
            cur = self._mouseKeys[btn]
            if mod is None:
                return cur[0] != InputAction.RELEASE
            else:
                return cur[0] != InputAction.RELEASE and cur[1] == mod
    def GetMouseBtnDown(self, btn: MouseButton, mod: InputModifier = None):
        '''Check if mouse btn is pressed down. If mod is not None, any modifier will be accepted.'''
        if btn not in self._mouseKeys:
            return False
        else:
            cur = self._mouseKeys[btn]
            if mod is None:
                return cur[0] == InputAction.PRESS
            else:
                return cur[0] == InputAction.PRESS and cur[1] == mod
    def GetMouseBtnUp(self, btn: MouseButton):
        '''Check if mouse btn is released.'''
        if btn not in self._mouseKeys:
            return False
        else:
            return self._mouseKeys[btn][0] == InputAction.RELEASE
    def GetMouseBtnHold(self, btn: MouseButton, mod: InputModifier = None):
        '''Check if mouse btn is hold. If mod is not None, any modifier will be accepted.'''
        if btn not in self._mouseKeys:
            return False
        else:
            cur = self._mouseKeys[btn]
            if mod is None:
                return cur[0] == InputAction.HOLD
            else:
                return cur[0] == InputAction.HOLD and cur[1] == mod
    def GetKey(self, key: Key, mod: InputModifier = None):
        '''Check if key is pressed. If mod is not None, any modifier will be accepted.'''
        if key not in self._keyboardKeys:
            return False
        else:
            cur = self._keyboardKeys[key]
            if mod is None:
                return cur[0] != InputAction.RELEASE
            else:
                return cur[0] != InputAction.RELEASE and cur[1] == mod
    def GetKeyDown(self, key: Key, mod: InputModifier = None):
        '''Check if key is pressed down. If mod is not None, any modifier will be accepted.'''
        if key not in self._keyboardKeys:
            return False
        else:
            cur = self._keyboardKeys[key]
            if mod is None:
                return cur[0] == InputAction.PRESS
            else:
                return cur[0] == InputAction.PRESS and cur[1] == mod
    def GetKeyUp(self, key: Key):
        '''Check if key is released.'''
        if key not in self._keyboardKeys:
            return False
        else:
            return self._keyboardKeys[key][0] == InputAction.RELEASE
    def GetKeyHold(self, key: Key, mod: InputModifier = None):
        '''Check if key is hold. If mod is not None, any modifier will be accepted.'''
        if key not in self._keyboardKeys:
            return False
        else:
            cur = self._keyboardKeys[key]
            if mod is None:
                return cur[0] == InputAction.HOLD
            else:
                return cur[0] == InputAction.HOLD and cur[1] == mod
    # endregion

__all__ = ['InputManager']