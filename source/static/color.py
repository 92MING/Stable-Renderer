import numpy as np
from .data_types.vector import Vector


class Color:
    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.r, self.g, self.b, self.a = r, g, b, a
        self._tidy_rgba()

    @staticmethod
    def from_vec(vector: Vector):
        if len(vector) == 3:
            return Color(*vector[:3], a=1.0)
        elif len(vector) == 4:
            return Color(*vector[:4])
        else:
            raise ValueError('Vector must have 3 or 4 elements')

    @staticmethod
    def from_int(r, g, b, a=255):
        return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def _tidy_rgba(self):
        self.r = max(0.0, min(1.0, self.r))
        self.g = max(0.0, min(1.0, self.g))
        self.b = max(0.0, min(1.0, self.b))
        self.a = max(0.0, min(1.0, self.a))

    @property
    def rgb(self):
        return np.array([self.r, self.g, self.b])

    @property
    def rgba(self):
        return np.array([self.r, self.g, self.b, self.a])

    @property
    def argb(self):
        return np.array([self.a, self.r, self.g, self.b])

    def get_hsv(self):
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

        return Vector(h, s, v)

    def set_color_from_hsv(self, hsv: Vector, a=1.0):
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

    def __eq__(self, color):
        return self.r == color.r and self.g == color.g and self.b == color.b and self.a == color.a

    def __ne__(self, color):
        return not self == color

    def __add__(self, color):
        return Color(self.r + color.r, self.g + color.g, self.b + color.b, self.a + color.a)

    def __sub__(self, color):
        return Color(self.r - color.r, self.g - color.g, self.b - color.b, self.a - color.a)

    def __mul__(self, value):
        if isinstance(value, Color):
            return Color(self.r * value.r, self.g * value.g, self.b * value.b, self.a * value.a)
        elif isinstance(value, (float, int)):
            return Color(self.r * value, self.g * value, self.b * value, self.a * value)
        # Add cases for vec3 and vec4 if needed

    def __truediv__(self, value):
        if isinstance(value, Color):
            return Color(self.r / value.r, self.g / value.g, self.b / value.b, self.a / value.a)
        elif isinstance(value, (float, int)):
            return Color(self.r / value, self.g / value, self.b / value, self.a / value)
        # Add cases for vec3 and vec4 if needed

    def __iadd__(self, color):
        self.r += color.r
        self.g += color.g
        self.b += color.b
        self.a += color.a
        self._tidy_rgba()
        return self

    def __isub__(self, color):
        self.r -= color.r
        self.g -= color.g
        self.b -= color.b
        self.a -= color.a
        self._tidy_rgba()
        return self

    def __imul__(self, value):
        if isinstance(value, Color):
            self.r *= value.r
            self.g *= value.g
            self.b *= value.b
            self.a *= value.a
        elif isinstance(value, (float, int)):
            self.r *= value
            self.g *= value
            self.b *= value
            self.a *= value
        # Add cases for vec3 and vec4 if needed
        self._tidy_rgba()
        return self

    def __itruediv__(self, value):
        if isinstance(value, Color):
            self.r /= value.r
            self.g /= value.g
            self.b /= value.b
            self.a /= value.a
        elif isinstance(value, (float, int)):
            self.r /= value
            self.g /= value
            self.b /= value
            self.a /= value
        # Add cases for vec3 and vec4 if needed
        self._tidy_rgba()
        return self

    def __str__(self):
        return "Color(r={}, g={}, b={}, a={})".format(self.r, self.g, self.b, self.a)


class ConstColor:
    # constant colors
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
