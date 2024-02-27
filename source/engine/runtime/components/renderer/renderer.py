'''The base class of all renderers. A renderer is a component that can draw to screen. It contains a list of materials.'''

from typing import Iterable, Union, List, Tuple
from functools import reduce

from ...component import Component
from engine.static.material import Material


class Renderer(Component):
    def __init__(self, gameObj, enable=True,
                 materials:Union[Material, Iterable[Material]] = None):
        super().__init__(gameObj, enable)
        self._materials:List[Material] = []
        if materials is not None:
            self.addMaterial(materials)

    @property
    def material(self)->Material:
        '''Return the first material in the list. If the list is empty, return None.'''
        return self._materials[0] if len(self._materials) > 0 else None
    @material.setter
    def material(self, value):
        '''Set the first material in the list. If the list is empty, add the material to the list.'''
        if len(self._materials) == 0:
            self._materials.append(value)
        else:
            self._materials[0] = value

    @property
    def materials(self)->Tuple[Material, ...]:
        '''Return the materials list.'''
        return tuple(self._materials)

    def addMaterial(self, material:Union[Material, Iterable[Material]], duplicateCheck=False):
        '''
        Add a material to the list.
        :param material: The material to add (Can be a Material object or a list of Material objects).
        :param duplicateCheck: If True, check if the material is already in the list. If so, do not add.
        '''
        if isinstance(material, Iterable):
            for m in material:
                self.addMaterial(m, duplicateCheck)
        else:
            if duplicateCheck:
                self._materials.append(material) if material not in self._materials else None
            else:
                self._materials.append(material)

    def removeMaterial(self, material:Material):
        '''
        Remove a material from the list. If the material is not in the list, do nothing.
        If there are more than one same materials in the list, only remove the last one.
        :param material: The material to remove.
        '''
        for i in range(len(self._materials) - 1, -1, -1):
            if self._materials[i] == material:
                self._materials.pop(i)
                break

    def _draw(self):
        '''Draw to screen. Override this method to implement your own drawing logic. Usually add render task to RenderManager here'''
        raise NotImplementedError

    def _drawAvailable(self):
        '''Check if this renderer is available to draw. Override this method to implement your own logic.'''
        return len(self._materials) > 0 and reduce(lambda x, y: x and y, map(lambda x: x.drawAvailable, self._materials), True)

    def lateUpdate(self):
        '''Called after all components' update() method are called.'''
        if self._drawAvailable():
            self._draw()


__all__ = ['Renderer']