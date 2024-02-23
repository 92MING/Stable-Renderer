import glm
from typing import Optional, Union, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from runtime.gameObj import GameObject

from utils.decorator import Overload
from runtime.component import Component

class Transform(Component):
    '''
    Transform component is used to represent the position, rotation and scale of a gameObject.

    Note:
        * using Left-Handed coordinate system (same as Unity)
        * transform order = T(R(S(v)))
    '''

    Unique = True # only one transform component is allowed on one gameObject

    def __init__(self, gameObj:'GameObject', enable=True):
        if not enable:
            raise Exception('Transform can not be disabled.')
        super().__init__(gameObj)
        self._localPos = glm.vec3(0, 0, 0)
        self._localRot = glm.quat()
        self._localScale = glm.vec3(1, 1, 1)

    # region runtime stuff
    @property
    def enable(self):
        return True # transform can not be disabled

    @enable.setter
    def enable(self, value):
        if not value:
            raise Exception('Transform can not be disabled.')

    @property
    def parent(self)->Optional['Transform']:
        '''return the transform on parent gameObject. If this transform has no parent or parent has no transform, return None'''
        if self.gameObj.parent is not None and self.gameObj.parent.hasComponent(Transform):
            return self.gameObj.parent.transform
        return None
    # endregion

    # region rotation
    @property
    def localRotation(self):
        '''return local rotation in quaternion'''
        return self._localRot

    @localRotation.setter
    def localRotation(self, value: Union[glm.quat, glm.vec3, Sequence[Union[int, float]]]):
        '''
        Set local rotation.

        Accept:
            * glm.quat: set by quaternion
            * glm.vec3: set by euler angles (in degree)
        '''
        if isinstance(value, glm.vec3):
            value = glm.quat(glm.radians(value))

        elif isinstance(value, Sequence):
            for i in value:
                if not isinstance(i, (int, float)):
                    raise TypeError('localRotation must be glm.quat or Sequence of 3 numbers')
            value = glm.quat(glm.radians(glm.vec3(value)))

        if not isinstance(value, glm.quat):
            raise TypeError(f'localRotation must be glm.quat, but got {value}({type(value)})')
        self._localRot = value

    @property
    def rotation(self):
        '''return global rotation'''
        forward = self.forward
        return glm.quatLookAtLH(forward, self.up)

    @rotation.setter
    def rotation(self, value: Union[glm.quat, glm.vec3, Sequence[Union[int, float]]]):
        '''
        Set global rotation.

        Accept:
            * glm.quat: set by quaternion
            * glm.vec3: set by euler angles (in degree)
        '''
        if isinstance(value, glm.vec3):
            value = glm.quat(glm.radians(value))
        elif isinstance(value, Sequence):
            for i in value:
                if not isinstance(i, (int, float)):
                    raise TypeError('rotation must be glm.quat or Sequence of 3 numbers')
            value = glm.quat(glm.radians(glm.vec3(value)))

        if not isinstance(value, glm.quat):
            raise TypeError('rotation must be glm.quat')
        if not self.parent:
            self._localRot = value
        else:
            newForward = value * glm.vec3(0, 0, 1)
            newUp = value * glm.vec3(0, 1, 0)
            newForwardInParentSpace = self.parent.inverseTransformDirection(newForward)
            newUpInParentSpace = self.parent.inverseTransformDirection(newUp)
            self._localRot = glm.quatLookAtLH(newForwardInParentSpace, newUpInParentSpace)

    def setLocalRotationX(self, angle:float, radian=False):
        '''set local rotation around x axis'''
        if not radian:
            angle = glm.radians(angle)
        self.localRotation = glm.rotate(self.localRotation, angle, glm.vec3(1, 0, 0))

    def setLocalRotationY(self, angle:float, radian=False):
        '''set local rotation around y axis'''
        if not radian:
            angle = glm.radians(angle)
        self.localRotation = glm.rotate(self.localRotation, angle, glm.vec3(0, 1, 0))

    def setLocalRotationZ(self, angle:float, radian=False):
        '''set local rotation around z axis'''
        if not radian:
            angle = glm.radians(angle)
        self.localRotation = glm.rotate(self.localRotation, angle, glm.vec3(0, 0, 1))

    @Overload
    def rotate(self, rotation:glm.quat):
        '''rotate locally by a quaternion'''
        self.localRotation = self.localRotation * rotation

    @Overload
    def rotate(self, axis:glm.vec3, angle:float, radian=False):
        '''rotate by an axis and an angle (in local space)'''
        if not radian:
            angle = glm.radians(angle)
        self._localRot = glm.rotate(self._localRot, angle, axis)

    def rotateLocalX(self, angle:float, radian=False):
        '''rotate around local x axis'''
        self.rotate(glm.vec3(1, 0, 0), angle, radian)

    def rotateLocalY(self, angle:float, radian=False):
        '''rotate around local y axis'''
        self.rotate(glm.vec3(0, 1, 0), angle, radian)

    def rotateLocalZ(self, angle:float, radian=False):
        '''rotate around local z axis'''
        self.rotate(glm.vec3(0, 0, 1), angle, radian)

    def rotateAround(self, center: glm.vec3, axis: glm.vec3, angle: float, radian=False):
        '''rotate around a point'''
        if not radian:
            angle = glm.radians(angle)
        center_to_pos = self.position   # vector from center to global position
        newPos = glm.rotate(center_to_pos, angle, axis) + center
        self.position = newPos
        self.rotate(axis, angle, radian)
    # endregion

    # region direction
    @property
    def forward(self)->glm.vec3:
        '''return forward direction in terms of global space'''
        return self.transformDirection(glm.vec3(0, 0, 1))

    @property
    def forwardInParentSpace(self)->glm.vec3:
        '''return forward direction in terms of parent space'''
        return glm.vec3(0, 0, 1) * self.localRotation

    @property
    def up(self) -> glm.vec3:
        '''return local up directio in terms of global space'''
        return self.transformDirection(glm.vec3(0, 1, 0))

    @property
    def upInParentSpace(self) -> glm.vec3:
        '''return up direction in terms of parent space'''
        return glm.vec3(0, 1, 0) * self.localRotation

    @property
    def right(self) -> glm.vec3:
        '''return local right direction in terms of global space'''
        return self.transformDirection(glm.vec3(1, 0, 0))

    @property
    def rightInParentSpace(self) -> glm.vec3:
        '''return right direction in terms of parent space'''
        return glm.vec3(1, 0, 0) * self.localRotation

    @forward.setter
    def forward(self, forward:glm.vec3):
        '''set forward direction in terms of global space.'''
        if self.parent:
            newForwardInParentSpace = self.parent.inverseTransformDirection(forward)
        else:
            newForwardInParentSpace = glm.normalize(forward)
        up = glm.vec3(0, 1, 0)
        if glm.dot(self.up, up) < 0:
            up = -up
        self._localRot = glm.quatLookAtLH(newForwardInParentSpace, up)

    @up.setter
    def up(self, up):
        '''set local up direction in terms of global space.'''
        if self.parent:
            newUpInParentSpace = self.parent.inverseTransformDirection(up)
        else:
            newUpInParentSpace = glm.normalize(up)
        forward = glm.vec3(0, 0, 1)
        if glm.dot(self.forward, forward) < 0:
            forward = -forward
        self._localRot = glm.quatLookAtLH(forward, newUpInParentSpace)

    @right.setter
    def right(self, right):
        '''set local right direction in terms of global space.'''
        if self.parent:
            newRightInParentSpace = self.parent.inverseTransformDirection(right)
        else:
            newRightInParentSpace = glm.normalize(right)
        up = glm.vec3(0, 1, 0)
        if glm.dot(self.up, up) < 0:
            up = -up
        forward = glm.cross(newRightInParentSpace, up)
        self._localRot = glm.quatLookAtLH(forward, up)

    def lookAt(self, target:Union['Transform', 'GameObject', glm.vec3, Sequence]):
        '''
        Rotate transform to make forward direction point to globalTarget, and up direction point to up.

        Args:
            * target: the global target position/game object/transform to look at
        '''
        from runtime.gameObj import GameObject
        if isinstance(target, GameObject):
            target = target.transform.position
        elif isinstance(target, Transform):
            target = target.position
        newForward = target - self.position
        self.forward = newForward
    # endregion

    # region position
    @property
    def localPosition(self)->glm.vec3:
        '''return the position of this transform in local space'''
        return self._localPos

    @localPosition.setter
    def localPosition(self, value: Union[glm.vec3, Sequence[Union[int, float]]]):
        if not isinstance(value, glm.vec3):
            try:
                value = glm.vec3(value)
            except:
                raise TypeError('localPosition must be glm.vec3 or Sequence of 3 numbers')
        self._localPos = value

    @property
    def position(self)->glm.vec3:
        '''return the position of this transform in global space'''
        return self.transformPoint(glm.vec3(0, 0, 0))

    @position.setter
    def position(self, value: Union[glm.vec3, Sequence[Union[int, float]]]):
        '''set the position of this transform in global space'''
        if not isinstance(value, glm.vec3):
            try:
                value = glm.vec3(value)
            except:
                raise TypeError('position must be glm.vec3 or Sequence of 3 numbers')
        if self.parent:
            pos_in_parent_space = self.parent.inverseTransformPoint(value)
            self.localPosition = pos_in_parent_space
        else:
            self.localPosition = value
    # endregion

    # region scale
    @property
    def localScale(self)->glm.vec3:
        '''return the scale of this transform in local space'''
        return self._localScale

    @localScale.setter
    def localScale(self, value):
        if not isinstance(value, glm.vec3):
            try:
                value = glm.vec3(value)
            except:
                raise TypeError('localScale must be glm.vec3 or Sequence of 3 numbers')
        self._localScale = value

    @property
    def scale(self)->glm.vec3:
        '''return the scale of this transform in global space'''
        if self.parent is None:
            return self.localScale
        return (self.parent.globalScaleMatrix * glm.vec4(self.localScale, 1)).xyz

    @scale.setter
    def scale(self, value: Union[glm.vec3, Sequence[Union[int, float]]]):
        '''set the scale of this transform in global space'''
        if not isinstance(value, glm.vec3):
            try:
                value = glm.vec3(value)
            except:
                raise TypeError('scale must be glm.vec3 or Sequence of 3 numbers')
        if self.parent is None:
            self.localScale = value
        else:
            self.localScale = (self.parent.inverseGlobalScaleMatrix * glm.vec4(value, 1)).xyz
    # endregion

    # region transform
    @property
    def translationMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local position'''
        return glm.translate(self._localPos)

    @property
    def rotationMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local rotation'''
        return glm.mat4_cast(self._localRot)

    @property
    def scaleMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local scale'''
        return glm.scale(self._localScale)

    @property
    def transformMatrix(self)->glm.mat4:
        '''
        Return the local matrix that can transform a vector to local space.
        '''
        return self.translationMatrix * self.rotationMatrix * self.scaleMatrix

    @property
    def globalTransformMatrix(self)->glm.mat4:
        '''
        Return the matrix that can transform a vector to global space
        This is used as `Model Matrix` in OpenGL
        '''
        if self.parent is None:
            return self.transformMatrix
        return self.parent.globalTransformMatrix * self.transformMatrix

    @property
    def globalScaleMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to global scale'''
        if self.parent is None:
            return self.scaleMatrix
        return self.parent.globalScaleMatrix * self.scaleMatrix

    @property
    def inverseGlobalTransformMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector from global space to local space'''
        return glm.inverse(self.globalTransformMatrix)

    @property
    def inverseGlobalScaleMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector from global scale to local scale'''
        return glm.inverse(self.globalScaleMatrix)

    def transformPoint(self, localPos)->glm.vec3:
        '''
        Regards the point as a child and transform it from local space to global space.
        Same as Unity's `TransformPoint`
        '''
        return (self.globalTransformMatrix * glm.vec4(localPos, 1)).xyz

    def inverseTransformPoint(self, globalPos)->glm.vec3:
        '''
        Regards the point as a child and transform it from global space to local space.
        Same as Unity's `InverseTransformPoint`
        '''
        return (self.inverseGlobalTransformMatrix * glm.vec4(globalPos, 1)).xyz

    def transformDirection(self, localDir)->glm.vec3:
        '''transform a direction from local space to global space (return a normalized vector)'''
        return glm.normalize(self.globalTransformMatrix * glm.vec4(localDir, 0)).xyz

    def inverseTransformDirection(self, globalDir: glm.vec3)->glm.vec3:
        '''transform a direction from global space to local space (return a normalized vector)'''
        return glm.normalize(self.inverseGlobalTransformMatrix * glm.vec4(globalDir, 0)).xyz
    # endregion