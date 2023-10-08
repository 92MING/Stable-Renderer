import numpy as np
from static.data_types.vector import Vector
from typing import Sequence

def _getVal(val):
    if isinstance(val, int):
        val = val / 255.0
    if isinstance(val, float):
        val = max(0.0, min(1.0, val))
    else:
        raise TypeError('val must be int or float')
    return val

class Color(Vector[float]):
    def __new__(cls, *values):
        if len(values) == 1:
            if not isinstance(values[0], (Sequence, np.ndarray)):
                values = (values,)
            else:
                values = values[0]
        if not ((3<=len(values)<=4) or len(values)==1):
            raise ValueError('Color must have 1, 3 or 4 values')
        if type(values[0]) == int:
            values = [_getVal(val) for val in values]
        if len(values) == 1:
            values = tuple(values) * 3 + (1.0,)
        elif len(values) == 3:
            values = tuple(values) + (1.0,)
        return np.array(values, dtype=cls.type()).view(cls)

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
    def rgb(self)->Vector[float]:
        return Vector[self.type()](self[:3])
    def hsv(self):
        '''return hsv color in tuple(h, s, v)'''
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

        return Vector[self.type()](h, s, v)
    def set_from_hsv(self, hsv: Vector, a=1.0):
        h, s, v = hsv[:3]
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
        self.a = a
        self._tidy_rgba()

    def __repr__(self):
        return f"Color({self.r:.2f}, {self.g:.2f}, {self.b:.2f}, {self.a:.2f})"
    def __str__(self):
        return f"[{self.r:.2f}, {self.g:.2f}, {self.b:.2f}, {self.a:.2f}]"

BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
YELLOW = Color(255, 255, 0)
ORANGE = Color(255, 165, 0)
PURPLE = Color(128, 0, 128)
CYAN = Color(0, 255, 255)
MAGENTA = Color(255, 0, 255)
GRAY = Color(128, 128, 128)
CLEAR = Color(0, 0, 0, 0)