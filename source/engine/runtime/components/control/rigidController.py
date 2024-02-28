from ...component import Component
from ..physics import RigidBody

class RigidController(Component):
    
    RequireComponent = (RigidBody,)
    
    # TODO