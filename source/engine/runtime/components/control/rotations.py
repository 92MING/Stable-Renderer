from typing import Literal
from ...component import Component

class EqualIntervalRotation(Component):
    def __init__(self, 
                 gameObj, 
                 enable=True, 
                 interval: int = 18,
                 axis: Literal['x', 'y', 'z'] = 'y',
                 update_mode: Literal['fixed_update', 'update'] = 'update'):
        super().__init__(gameObj, enable)
        self.rotation_interval = 360 // interval
        self.rotation_axis = axis
        self.update_mode = update_mode
        
    def _update(self):
        match self.rotation_axis:
            case 'x':
                self.transform.rotateLocalX(self.rotation_interval)
            case 'y':
                self.transform.rotateLocalY(self.rotation_interval)
            case 'z':
                self.transform.rotateLocalZ(self.rotation_interval)
            case _:
                raise ValueError(f"Invalid axis: {self.rotation_axis}")

    def update(self):
        if self.update_mode == 'update':
            self._update()
            
    def fixedUpdate(self):
        if self.update_mode == 'fixed_update':
            self._update()

class AutoRotation(Component):
    def __init__(self, 
                 gameObj, 
                 enable=True, 
                 angular_spd: float = 4,
                 axis: Literal['x', 'y', 'z'] = 'y'):
        super().__init__(gameObj, enable)
        self.angular_spd = angular_spd
        self.rotation_axis = axis
    
    def update(self):
        match self.rotation_axis:
            case 'x':
                self.transform.rotateLocalX(self.angular_spd * self.engine.RuntimeManager.DeltaTime)
            case 'y':
                self.transform.rotateLocalY(self.angular_spd * self.engine.RuntimeManager.DeltaTime)
            case 'z':
                self.transform.rotateLocalZ(self.angular_spd * self.engine.RuntimeManager.DeltaTime)
            case _:
                raise ValueError(f"Invalid axis: {self.rotation_axis}")
            
            
            
__all__ = ['EqualIntervalRotation', 'AutoRotation']