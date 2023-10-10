import glm
from typing import cast

def _getVal(val):
    if isinstance(val, int):
        val = val / 255.0
    if isinstance(val, float):
        val = max(0.0, min(1.0, val))
    else:
        raise TypeError('val must be int or float')
    return val

class _classproperty:
    def __init__(self, func):
        self._func = func
    def __get__(self, cls, owner):
        return self._func(owner)

class Color(glm.vec4):

    def __new__(cls, r, g, b, a=1.0):
        r = _getVal(r)
        g = _getVal(g)
        b = _getVal(b)
        a = _getVal(a)
        return cast(Color, glm.vec4(r, g, b, a))

    # region constants
    @_classproperty
    def BLACK(cls):
        if not hasattr(cls, '_BLACK'):
            cls._BLACK = cls(0, 0, 0)
        return cls._BLACK
    @_classproperty
    def WHITE(cls):
        if not hasattr(cls, '_WHITE'):
            cls._WHITE = cls(255, 255, 255)
        return cls._WHITE
    @_classproperty
    def RED(cls):
        if not hasattr(cls, '_RED'):
            cls._RED = cls(255, 0, 0)
        return cls._RED
    @_classproperty
    def GREEN(cls):
        if not hasattr(cls, '_GREEN'):
            cls._GREEN = cls(0, 255, 0)
        return cls._GREEN
    @_classproperty
    def BLUE(cls):
        if not hasattr(cls, '_BLUE'):
            cls._BLUE = cls(0, 0, 255)
        return cls._BLUE
    @_classproperty
    def YELLOW(cls):
        if not hasattr(cls, '_YELLOW'):
            cls._YELLOW = cls(255, 255, 0)
        return cls._YELLOW
    @_classproperty
    def ORANGE(cls):
        if not hasattr(cls, '_ORANGE'):
            cls._ORANGE = cls(255, 165, 0)
        return cls._ORANGE
    @_classproperty
    def PURPLE(cls):
        if not hasattr(cls, '_PURPLE'):
            cls._PURPLE = cls(128, 0, 128)
        return cls._PURPLE
    @_classproperty
    def CYAN(cls):
        if not hasattr(cls, '_CYAN'):
            cls._CYAN = cls(0, 255, 255)
        return cls._CYAN
    @_classproperty
    def MAGENTA(cls):
        if not hasattr(cls, '_MAGENTA'):
            cls._MAGENTA = cls(255, 0, 255)
        return cls._MAGENTA
    @_classproperty
    def GRAY(cls):
        if not hasattr(cls, '_GRAY'):
            cls._GRAY = cls(128, 128, 128)
        return cls._GRAY
    @_classproperty
    def CLEAR(cls):
        if not hasattr(cls, '_CLEAR'):
            cls._CLEAR = cls(0, 0, 0, 0)
        return cls._CLEAR
    # endregion

    def _tidy_rgba(self):
        self.r = _getVal(self.r)
        self.g = _getVal(self.g)
        self.b = _getVal(self.b)
        self.a = _getVal(self.a)

    @property
    def r(self):
        return self[0]
    @r.setter
    def r(self, val):
        self[0] = _getVal(val)
    @property
    def g(self):
        return self[1]
    @g.setter
    def g(self, val):
        self[1] = _getVal(val)
    @property
    def b(self):
        return self[2]
    @b.setter
    def b(self, val):
        self[2] = _getVal(val)
    @property
    def a(self):
        return self[3]
    @a.setter
    def a(self, val):
        self[3] = _getVal(val)
    @property
    def rgb(self):
        return self.xyz
    def hsv(self):
        '''return hsv color in glm.vec4(h, s, v, a)'''
        h, s, v = 0.0, 0.0, 0.0
        max_val = max(self.r, self.g, self.b)
        min_val = min(self.r, self.g, self.b)

        if max_val == min_val:
            h = 0
        elif max_val == self.r:
            h = 60 * (self.g - self.b) / (max_val - min_val)
        elif max_val == self.g:
            h = 60 * (self.b - self.r) / (max_val - min_val) + 120
        elif max_val == self.b:
            h = 60 * (self.r - self.g) / (max_val - min_val) + 240
        if h < 0:
            h += 360
        s = 0 if max_val == 0 else (max_val - min_val) / max_val
        v = max_val
        return glm.vec4(h, s, v, self.a)
    def set_from_hsv(self, hsv: glm.vec3, alpha: float = 1.0):
        h, s, v = hsv.x, hsv.y, hsv.z
        if s == 0:
            self.r, self.g, self.b = v, v, v
        else:
            i = int(h / 60)
            f = h / 60 - i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))

            if i == 0:
                self.r, self.g, self.b = v, t, p
            elif i == 1:
                self.r, self.g, self.b = q, v, p
            elif i == 2:
                self.r, self.g, self.b = p, v, t
            elif i == 3:
                self.r, self.g, self.b = p, q, v
            elif i == 4:
                self.r, self.g, self.b = t, p, v
            elif i == 5:
                self.r, self.g, self.b = v, p, q
        self.a = alpha
        self._tidy_rgba()

__all__ = ['Color']