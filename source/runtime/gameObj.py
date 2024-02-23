'''GameObject is the class contains components and behaviors.'''

from typing import Union, List, Optional, Set, Dict
import heapq
import glm

from utils.base_clses import NamedObj
from utils.global_utils import GetOrAddGlobalValue
from runtime.component import Component, ComponentMeta
from runtime.components.transform import Transform
from runtime.engineObj import EngineObj

class AutoSortList(list):
    def append(self, obj):
        heapq.heappush(self, obj)
    def extend(self, objs):
        for obj in objs:
            self.append(obj)

_gameObj_tags: Dict[str, Set['GameObject']] = GetOrAddGlobalValue("_GAMEOBJ_TAGS", {})
'''{tag: set[gameObj]}'''

_root_gameObjs: List['GameObject'] = GetOrAddGlobalValue("_ROOT_GAMEOBJS", []) # list of root gameObj(no parent)
'''list of root gameObj. Root gameObjs are gameObjs without parent'''


class GameObject(EngineObj, NamedObj):

    # region class methods
    @staticmethod
    def RootObjs()->List['GameObject']:
        return _root_gameObjs

    @staticmethod
    def FindObjs_ByTag(tag)->set:
        return _gameObj_tags.get(tag, set())

    @classmethod
    def FindObj_ByName(cls, name)->'GameObject':
        '''Return None if not found'''
        return cls.GetInstance(name)
    # endregion

    def __init__(self,
                 name:str,
                 active:bool=True,
                 parent:'GameObject'=None,
                 tags:Union[str, list, tuple, set]=None,
                 position:Union[list, tuple, glm.vec3]=None,
                 rotation:Union[list, tuple, glm.vec3]=None,
                 scale:Union[list, tuple, glm.vec3]=None,
                 needTransform=True):
        '''
        GameObject is the class contains components.
        :param name: name of this gameObj. Can be used to find this gameObj. Cannot be None or duplicated.
        :param active: whether this gameObj is active. If not active, this gameObj and all its children will not be updated.
        :param parent: parent gameObj. If None, this gameObj will be added to root gameObj list.
        :param tags: tags of this gameObj. Can be used to find this gameObj. Can be None, str, list, tuple or set.
        :param position: local position of this gameObj. If needTransform is False, error will be raised if posiiton is not None.
        :param rotation: local rotation of this gameObj in euler angles. If needTransform is False, error will be raised if rotation is not None.
        :param scale: local scale of this gameObj. If needTransform is False, error will be raised if scale is not None.
        :param needTransform: whether this gameObj need a transform component.
        '''
        super().__init__(name)
        self._active = active
        self._tags = set()
        self._parent = parent
        self._children = []
        self._components:List[Component] = AutoSortList()
        if parent is None:
            _root_gameObjs.append(self)
        else:
            self.parent.children.append(self) if self not in self.parent.children else None
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.addTag(tag)
        if needTransform:
            self.addComponent(Transform)
        if not needTransform and (position is not None or rotation is not None or scale is not None):
            raise ValueError("If needTransform is False, posiiton, rotation and scale cannot be set.")
        if position is not None:
            self.transform.localPosition = glm.vec3(position)
        if rotation is not None:
            self.transform.localRotation = glm.vec3(rotation[0], rotation[1], rotation[2])
        if scale is not None:
            self.transform.localScale = glm.vec3(scale)

    # region child & parent
    def hasChild(self, child):
        return child in self.children

    def getChild(self, index):
        return self.children[index]

    @property
    def children(self)->List['GameObject']:
        return self._children

    @property
    def childCount(self):
        return len(self.children)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        """
        Set parent object. If `new_parent` is None, this object will be moved to root object list.
        """
        if new_parent == self.parent:
            return
        if self.parent is not None:
            self.parent.children.remove(self)
        else:
            GameObject.RootObjs().remove(self)
        self._parent = new_parent
        if self.parent is not None:
            self.parent.children.append(self)
        else:
            GameObject.RootObjs().append(self)

    @property
    def siblingIndex(self):
        if self.parent is not None:
            return self.parent.children.index(self)
        else:
            return GameObject.RootObjs().index(self)

    @siblingIndex.setter
    def siblingIndex(self, new_index):
        if new_index == self.siblingIndex:
            return
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent.children.insert(new_index, self)
        else:
            GameObject.RootObjs().remove(self)
            GameObject.RootObjs().insert(new_index, self)
    # endregion

    # region active & destroy
    @property
    def active(self):
        active = self._active
        par = self.parent
        while par is not None and active:
            active = par.active
            par = par.parent
        return active

    @active.setter
    def active(self, value:bool):
        if self._active == value:
            return
        self._active = value
        for comp in self.allComponents(enableOnly=True):
            if not value:
                comp.onDisable()
            else:
                comp.onEnable()

    def destroy(self):
        if self.parent is not None:
            self.parent.children.remove(self)
        else:
            GameObject.root_game_objects.remove(self)
        for att_list in self.attributes.values():
            for att in att_list:
                att.destroy()
        for child in list(self.children):
            child.destroy()
    # endregion

    # region tags
    @property
    def tags(self):
        return self._tags

    def hasTag(self, tag):
        return tag in self.tags

    def addTag(self, tag):
        if self.hasTag(tag):
            return
        self.tags.add(tag)
        if tag not in _gameObj_tags:
            _gameObj_tags[tag] = set()
        _gameObj_tags[tag].add(self)

    def removeTag(self, tag):
        if not self.hasTag(tag):
            return
        self.tags.remove(tag)
        _gameObj_tags[tag].remove(self)
    # endregion

    # region components
    def components(self, enableOnly=True)->List[Component]:
        '''
        return components it has
        :param enableOnly: only return enabled components
        :return:
        '''
        if enableOnly:
            return [comp for comp in self._components if comp.enable]
        else:
            return self._components

    def allComponents(self, enableOnly=True)->List[Component]:
        '''
        return all components it and its children have
        :param enableOnly: only return enabled components
        '''
        result = AutoSortList()
        for child in self.children:
            result.extend(child.childComponents(enableOnly))
        result.extend(self.components(enableOnly))
        return result

    def hasComponent(self, comp: Union[ComponentMeta, str, Component], enableOnly=False):
        '''
        whether it has component of type `comp`. If there exist a subclass of `comp`, also return True
        :param comp: Subclass of Component / component instance / component name
        :param enableOnly: whether only search enabled components
        :return: bool
        '''
        if isinstance(comp, ComponentMeta):
            return any(isinstance(c, comp) for c in self.components(enableOnly=enableOnly))
        elif isinstance(comp, str):
            return any(c.ComponentName == comp for c in self.components(enableOnly=enableOnly))
        elif isinstance(comp, Component):
            return comp in self.components(enableOnly=enableOnly)
        else:
            raise TypeError("comp must be a subclass of Component, or a component instance, or component name")

    def getComponent(self, comp: Union[ComponentMeta, str, Component], enableOnly=False)->Optional[Component]:
        '''
        return the first component of type `comp`
        :param comp: Subclass of Component / component instance / component name
        :param enableOnly: whether only search enabled components
        :return: component or None if not found
        '''
        if not isinstance(comp, (ComponentMeta, str, Component)):
            raise TypeError("comp must be a subclass of Component, or a component instance, or component name")
        if isinstance(comp, str):
            for c in self.components(enableOnly=enableOnly):
                if c.ComponentName == comp:
                    return c
        elif isinstance(comp, Component):
            for c in self.components(enableOnly=enableOnly):
                if c == comp:
                    return c
        else:
            for c in self.components(enableOnly=enableOnly):
                if isinstance(c, comp):
                    return c
        return None

    def getComponents(self, comp: Union[ComponentMeta, str, Component], enableOnly=False):
        '''
        return all components of type `comp`
        :param comp: Subclass of Component / component instance / component name
        :param enableOnly: whether only search enabled components
        :return: list of components
        '''
        if not isinstance(comp, (ComponentMeta, str, Component)):
            raise TypeError("comp must be a subclass of Component, or a component instance, or component name")
        if isinstance(comp, ComponentMeta):
            return [c for c in self.components(enableOnly=enableOnly) if isinstance(c, comp)]
        elif isinstance(comp, str):
            return [c for c in self.components(enableOnly=enableOnly) if c.ComponentName == comp]
        elif isinstance(comp, Component):
            return [c for c in self.components(enableOnly=enableOnly) if c == comp]

    def removeComponent(self, comp: Union[ComponentMeta, str, Component], enableOnly=False):
        '''
        remove the first component of type `comp`
        :param comp: Subclass of Component / component instance / component name
        '''
        if not isinstance(comp, (ComponentMeta, str, Component)):
            raise TypeError("comp must be a subclass of Component, or a component instance, or component name")
        if isinstance(comp, str):
            for c in self.components(enableOnly=enableOnly):
                if c.ComponentName == comp:
                    c.onDisable()
                    c.onDestory()
                    return
        elif isinstance(comp, Component):
            for c in self.components(enableOnly=enableOnly):
                if c == comp:
                    c.onDisable()
                    c.onDestory()
                    return
        else:
            for c in self.components(enableOnly=enableOnly):
                if isinstance(c, comp):
                    c.onDisable() # call onDisable before destroy
                    c.onDestory()
                    return

    def removeComponents(self, comp: Union[ComponentMeta, str, Component], enableOnly=False):
        '''
        remove all components of type `comp`
        :param comp: Subclass of Component / component instance / component name
        '''
        if not isinstance(comp, (ComponentMeta, str, Component)):
            raise TypeError("comp must be a subclass of Component")
        if isinstance(comp, str):
            for c in self.components(enableOnly=enableOnly):
                if c.ComponentName == comp:
                    c.onDisable()
                    c.onDestory()
        elif isinstance(comp, Component):
            for c in self.components(enableOnly=enableOnly):
                if c == comp:
                    c.onDisable()
                    c.onDestory()
        else:
            for c in self.components(enableOnly=enableOnly):
                if isinstance(c, comp):
                    c.onDisable() # call onDisable before destroy
                    c.onDestory()

    def addComponent(self, comp: Union[ComponentMeta, str, Component], enable=True, *args, **kwargs)->Optional[Component]:
        '''
        add a component of type `comp`
        :param comp: Subclass of Component / component instance / component name
        :param enable: whether enable the component
        :param args: args for the constructor of `comp`
        :param kwargs: kwargs for the constructor of `comp`
        :return: the component added
        '''
        if not isinstance(comp, (ComponentMeta, str, Component)):
            raise TypeError("comp must be a subclass of Component")
        if isinstance(comp, (ComponentMeta, str)):
            comp = Component.FindComponentCls(comp) if isinstance(comp, str) else comp
            if comp.Unique and self.hasComponent(comp):
                return None # already has a unique component
            for require in comp.RequireComponent:
                if not self.hasComponent(require):
                    newComp = self.addComponent(require, enable=enable, )
                    if newComp is None:
                        raise RuntimeError(f"GameObject: {self.name} has no component: {require}. Auto add failed")
            c = comp(self, enable, *args, **kwargs)
            self._components.append(c)
            c._tryAwake() # call awake if the gameobject is active and the component is enabled
            return c
        else: # comp is a component instance
            if comp.gameObj != self:
                raise RuntimeError(f"Component: {comp} is not belong to GameObject: {self.name}. Can't add")
            if comp.Unique and self.hasComponent(comp):
                return None
            for require in comp.RequireComponent:
                if not self.hasComponent(require):
                    newComp = self.addComponent(require, enable=enable, )
                    if newComp is None:
                        raise RuntimeError(f"GameObject: {self.name} has no component: {require}. Auto add failed")
            self._components.append(comp)
            comp._tryAwake()  # call awake if the gameobject is active and the component is enabled

    @property
    def transform(self) -> Transform:
        t = self.getComponent(Transform)
        if t is None:
            raise RuntimeError(f"GameObject: {self.name} has no Transform")
        return t
    # endregion

    # region internal use
    def _fixedUpdate(self):
        '''internal use for fixedUpdate and components'''
        comps = self.components(enableOnly=True)
        for comp in comps:
            comp.fixedUpdate()
        for child in self.children:
            child._fixedUpdate()

    def _lateUpdate(self):
        '''internal use for lateUpdate and components'''
        comps = self.components(enableOnly=True)
        for comp in comps:
            comp.lateUpdate()
        for child in self.children:
            child._lateUpdate()

    def _update(self):
        '''internal use for update and components'''
        comps = self.components(enableOnly=True)
        for comp in comps:
            comp.update()
        for child in self.children:
            child._update()

    @staticmethod
    def _RunFixedUpdate():
        '''internal use for fixedUpdate and components'''
        for obj in _root_gameObjs:
            obj._fixedUpdate()

    @staticmethod
    def _RunLateUpdate():
        '''internal use for lateUpdate and components'''
        for obj in _root_gameObjs:
            obj._lateUpdate()

    @staticmethod
    def _RunUpdate():
        '''internal use for update and components'''
        for obj in _root_gameObjs:
            obj._update()
    # endregion


__all__ = ['GameObject']