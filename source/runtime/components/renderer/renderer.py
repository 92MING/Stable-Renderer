from runtime.component import Component
from static.material import Material

class Renderer(Component):
    def __init__(self, gameObj, enable=True, material: Material = None):
        super().__init__(gameObj, enable)
        self._material = material

    @property
    def material(self)->Material:
        return self._material
    @material.setter
    def material(self, value):
        self._material = value

    def _draw(self):
        '''Draw to screen. Override this method to implement your own drawing logic.'''
        raise NotImplementedError
    def _drawAvailable(self):
        '''Check if this renderer is available to draw. Override this method to implement your own logic.'''
        return self._material is not None and self._material.drawAvailable

    def lateUpdate(self):
        '''Called after all components' update() method are called.'''
        if self._drawAvailable():
            self._draw()


__all__ = ['Renderer']