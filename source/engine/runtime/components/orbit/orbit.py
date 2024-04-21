from typing import TYPE_CHECKING
import glm

from ...component import Component


class CircularOrbit(Component):
    def __init__(self,
                gameObject,
                enable: bool = True,
                radius: float = 5, theta: float = 0, phi: float = 0,
                theta_speed: float = 10, phi_speed: float = 0.138, 
                **kwargs):
        """
        A component that makes the gameObject orbit around the origin in a circular path.

        Args:
            gameObject (GameObject): A reference to the gameObject that this component is attached to.
            enable (bool, optional): Whether to execute this component or not. Defaults to True.
            radius (float, optional): The radius of orbiting and distance from (0,0,0). Defaults to 5.
            theta (float, optional): The xz plane angle. Increasing theta causes the object orbit 
                in clockwise motion, if your camera is at positive y positions. Defaults to 0.
            phi (float, optional): The y axis angle. Increasing phi causes the object orbit in 
                vertical motion. Defaults to 0.
            theta_speed (float, optional): Change of theta position per update call. Defaults to 10.
            phi_speed (float, optional): Change of phi position per update call. Defaults to 0.138.
        """
        super().__init__(gameObject, enable, **kwargs)
        self.radius = radius
        self.theta = theta
        self.phi = phi

        self.theta_speed = theta_speed
        self.phi_speed = phi_speed

    def start(self):
        self.transform.position = glm.vec3(
            self.radius * glm.sin(glm.radians(self.phi)) * glm.cos(glm.radians(self.theta)),
            self.radius * glm.cos(glm.radians(self.phi)),
            self.radius * glm.sin(glm.radians(self.phi)) * glm.sin(glm.radians(self.theta))
        )
        self.transform.lookAt(glm.vec3(0, 0, 0))
        
    def update(self):
        self.phi += self.phi_speed
        self.theta += self.theta_speed
        self.transform.position = glm.vec3(
            self.radius * glm.sin(glm.radians(self.phi)) * glm.cos(glm.radians(self.theta)),
            self.radius * glm.cos(glm.radians(self.phi)),
            self.radius * glm.sin(glm.radians(self.phi)) * glm.sin(glm.radians(self.theta))
        )
        self.transform.lookAt(glm.vec3(0, 0, 0))


class HelicalOrbit(CircularOrbit):
    def __init__(self,
                 gameObject,
                 enable: bool = True,
                 radius: float = 5, theta: float = 0, phi: float = 0,
                 theta_speed: float = 10, phi_speed_per_full_theta: float = 50, 
                 history_length: int = 1000,
                 **kwargs):
        """
        Initialize the Orbit component.

        Args:
            gameObject: The game object to which this component is attached.
            enable: Whether the component is enabled or not. Default is True.
            radius: The radius of the orbit. Default is 5.
            theta: The initial theta angle of the orbit. Default is 0.
            phi: The initial phi angle of the orbit. Default is 0.
            theta_speed: The speed at which the theta angle changes. Default is 10.
            phi_speed_per_full_theta: The speed at which the phi angle changes per 
                full theta rotation. Default is 50.
            history_length: The length of the history buffer. Default is 1000.
            **kwargs: Additional keyword arguments.

        """
        converted_phi_speed = (theta_speed / 360) * phi_speed_per_full_theta
        super().__init__(gameObject,
                         enable,
                         radius, theta, phi,
                         theta_speed, converted_phi_speed,
                         **kwargs)
        self.phi_speed_per_full_theta = phi_speed_per_full_theta

