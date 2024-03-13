import glm
from typing import ClassVar, TYPE_CHECKING
if TYPE_CHECKING:
    from engine.managers import RuntimeManager
    from ...gameObj import GameObject
    
from utils.decorators import Overload
from ...component import Component


class RigidBody(Component):
    
    _RuntimeManager: ClassVar['RuntimeManager'] = None
    
    @classmethod
    def _GetRuntimeManager(cls):
        if not cls._RuntimeManager:
            cls._RuntimeManager = cls.engine.RuntimeManager
        return cls._RuntimeManager    
        
    velocity: glm.vec3 = glm.vec3(0)
    '''velocity in unit per second'''
    angularVelocity: glm.vec3 = glm.vec3(0)
    '''angular velocity in degrees per second'''
    _mass: float = 1
    '''mass of this body. Will affect the force applied to this body. 1 means 1 unit per second^2 will move this body 1 unit per second. 0 means this body is static. Negative value is not allowed.'''
    isKinematic: bool = True
    '''if true, this body will not be affected by any force.'''
    useGravity: bool = False
    '''if true, this body will be affected by gravity. If isKinematic is false, this property will be ignored.'''
    
    def __init__(self, 
                 gameObj: 'GameObject', 
                 enable=True, 
                 mass: float = 1,
                 velocity: glm.vec3 = glm.vec3(0),
                 angularVelocity: glm.vec3 = glm.vec3(0),
                 isKinematic: bool = True,
                 useGravity: bool = False):
        super().__init__(gameObj, enable)
        self.mass = mass
        self.velocity = velocity
        self.angularVelocity = angularVelocity
        self.isKinematic = isKinematic
        self.useGravity = useGravity
    
    # region property
    @property
    def mass(self):
        return self._mass
    @mass.setter
    def mass(self, value: float):
        if value < 0:
            raise ValueError('mass cannot be negative')
        self._mass = value
    # endregion
    
    def fixedUpdate(self):
        if self.isKinematic:
            if self.useGravity:
                self.velocity += glm.vec3(0, -9.8, 0) / 2
            self.transform.position += self.velocity
            self.transform.rotate(self.angularVelocity.x, self.angularVelocity.y, self.angularVelocity.z)
    
    @Overload
    def addForce(self, force: glm.vec3):
        '''force in unit per second^2'''
        self.velocity += force / self.mass
        
    @Overload
    def addForce(self, direction: glm.vec3, magnitude: float):
        '''direction is a unit vector, magnitude is in unit per second^2'''
        self.velocity += direction * magnitude / self.mass
        
    # TODO