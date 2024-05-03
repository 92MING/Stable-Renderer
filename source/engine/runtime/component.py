from engine.runtime.base_clses import EngineObj
from common_utils.global_utils import GetOrAddGlobalValue
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gameObj import GameObject
    from .components.transform import Transform

__ENGINE_COMPONENT_CLSES__ = GetOrAddGlobalValue("__ENGINE_COMPONENT_CLSES__", {})

class ComponentMeta(type):
    '''
    This metaclass is used to make sure that there is only one cls for each component.
    '''
    def __new__(cls, *args, **kwargs):
        clsname = args[0]
        if clsname in __ENGINE_COMPONENT_CLSES__:
            return __ENGINE_COMPONENT_CLSES__[clsname]      # type: ignore
        else:
            cls = super().__new__(cls, *args, **kwargs)
            __ENGINE_COMPONENT_CLSES__[clsname] = cls   # type: ignore
            return cls


class Component(EngineObj, metaclass=ComponentMeta):
    '''Base cls of all components, e.g. Camera, Light, etc.'''

    # region cls properties
    Priority = 0
    '''
    Override this property to change the priority of this component.
    Lower value means higher priority.
    Note that the priority of a component is only for the components in the same GameObject.
    '''

    Unique = False
    '''
    Override this property to change the uniqueness of this component.
    If this property is True, there can be only one component of this type in a GameObject.
    '''

    RequireComponent = ()
    '''
    Override this property to change the required components of this component. You can input Component cls or str(cls name).
    e.g. Camera.RequireComponent = (Transform, )
    '''

    @classmethod
    def ComponentName(cls):
        return cls.__qualname__

    @classmethod
    def FindComponentCls(cls, cls_name):
        try:
            return __ENGINE_COMPONENT_CLSES__[cls_name]     # type: ignore
        except KeyError:
            raise KeyError(f'Component with class name: {cls_name} not found.')

    def __class_getitem__(cls, item):
        '''You can find a component cls by using this syntax: Component[cls_name]'''
        return cls.FindComponentCls(item)
    # endregion

    def __init__(self, gameObj:'GameObject', enable=True, **kwargs):
        '''Each component must start with super().__init__(gameObj, enable, ...)'''
        if self.__class__.__qualname__ == 'Component':
            raise Exception('Component is an abstract class. You can not create an instance of it.')
        if gameObj is None:
            raise Exception('gameObj can not be None.')
        self._gameObj = gameObj
        self._enable = enable
        self._started = False
        self._awaked = False

    def __ge__(self, other):
        return self.Priority >= other.Priority

    def __gt__(self, other):
        return self.Priority > other.Priority

    def __le__(self, other):
        return self.Priority <= other.Priority

    def __lt__(self, other):
        return self.Priority < other.Priority

    def awake(self):
        '''Called when this component is added to a GameObject (in case it is enabled). Awake does not follow the priority rule & will be called immediately.'''
        pass

    def start(self):
        '''Called when this component runs for the first time'''
        pass

    def _tryAwake(self):
        if self.enable:
            if not self._awaked :
                self._awaked = True
                self.awake()
            self.onEnable()

    def _tryStart(self):
        if not self._started and self.enable and self._awaked:
            self._started = True
            self.start()

    def _checkAwakeAndStart(self):
        self._tryAwake()
        self._tryStart()

    def fixedUpdate(self):
        '''Called every fixed frame'''
        self._checkAwakeAndStart()

    def update(self):
        '''Called every frame'''
        self._checkAwakeAndStart()

    def lateUpdate(self):
        '''Called every frame after update'''
        self._checkAwakeAndStart()

    def onEnable(self):
        '''Called when this component is enabled. onEnable will not follow the priority rule.'''
        pass

    def onDisable(self):
        '''
        Called when this component is disabled. onDisable will not follow the priority rule.
        When destroy a GameObject, onDisable will be called before onDestroy.
        '''

    def onDestroy(self):
        '''
        Called when this component is destroyed.
        Component can only be destroyed by GameObject.removeComponent
        onDestroy will not follow the priority rule.
        onDisable will be called before onDestroy.
        '''

    @property
    def gameObj(self)->'GameObject':
        return self._gameObj

    @property
    def transform(self)->'Transform':
        '''
        Shortcut to get the transform of the GameObject that this component is attached to.
        Note that gameobj can have no transform, it will rise an error in that case.
        '''
        return self._gameObj.transform

    @property
    def enable(self):
        return self._enable and self._gameObj.active

    @enable.setter
    def enable(self, value):
        self.setEnable(value)

    def setEnable(self, value):
        if self._enable != value:
            self._enable = value
            if self._gameObj.active:
                if value:
                    self._tryAwake()
                else:
                    self.onDisable()



__all__ = ['Component']