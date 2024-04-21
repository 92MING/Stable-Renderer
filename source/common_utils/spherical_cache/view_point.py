import math
import torch
from pydantic import BaseModel


class ViewPoint(BaseModel):
    radius: float
    phi: float  # inclination
    theta: float  # azimuth

    def to_tensor(self):
        return torch.tensor([self.radius, self.phi, self.theta], dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(radius=tensor[0].item(), phi=tensor[1].item(), theta=tensor[2].item())

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float):
        radius = math.sqrt(x**2 + y**2 + z**2)
        theta = math.atan2(z, x)
        phi = math.acos(y / radius)
        return cls(radius=radius, phi=phi, theta=theta)

    def to_cartesian(self):
        x = self.radius * math.sin(self.phi) * math.cos(self.theta)
        y = self.radius * math.cos(self.phi)
        z = self.radius * math.sin(self.phi) * math.sin(self.theta)
        return x, y, z

    def __str__(self):
        return f'radius: {self.radius}, phi: {self.phi}, theta: {self.theta}'

    def __repr__(self):
        return self.__str__()
