from runtime.component import Component
import glm
from typing import Optional, Union, Sequence

class Transform(Component):
    '''記載物件的位置、旋轉、縮放等資訊、操作'''

    Unique = True # 一個物件只能有一個Transform

    def __init__(self, gameObj:'GameObject', enable=True):
        if not enable:
            raise Exception('Transform can not be disabled.')
        super().__init__(gameObj)
        self.localPos = glm.vec3(0, 0, 0)
        self._localRot = glm.quat() # rotation cannot be set directly since it is quaternion
        self.localScale = glm.vec3(1, 1, 1)
    @property
    def enable(self):
        return True
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

    # region rotation
    def localRot(self, radian=False)->glm.vec3:
        if not radian:
            return glm.degrees(glm.eulerAngles(self._localRot))
        else:
            return glm.eulerAngles(self._localRot)
    def setLocalRot(self, x, y, z, radian=False):
        if not radian:
            x, y, z = glm.radians(x), glm.radians(y), glm.radians(z)
        self._localRot = glm.quat(glm.vec3(x, y, z))
    def localRotX(self, radian=False)->float:
        return self.localRot(radian)[0]
    def localRotY(self, radian=False)->float:
        return self.localRot(radian)[1]
    def localRotZ(self, radian=False)->float:
        return self.localRot(radian)[2]
    def setLocalRotX(self, x, radian=False):
        localRot = self.localRot(radian)
        self.setLocalRot(x, localRot[1], localRot[2], radian)
    def setLocalRotY(self, y, radian=False):
        localRot = self.localRot(radian)
        self.setLocalRot(localRot[0], y, localRot[2], radian)
    def setLocalRotZ(self, z, radian=False):
        localRot = self.localRot(radian)
        self.setLocalRot(localRot[0], localRot[1], z, radian)

    def globalRot(self, radian=False)->glm.vec3:
        gloablForward = self.forward
        globalUp = self.up
        if not radian:
            return glm.degrees(glm.eulerAngles(glm.quatLookAt(gloablForward, globalUp)))
        else:
            return glm.eulerAngles(glm.quatLookAt(gloablForward, globalUp))

    def rotateLocalAxis(self, axis:glm.vec3, angle:float, radian=False):
        if not radian:
            angle = glm.radians(angle)
        angle = -angle
        self._localRot = glm.rotate(self._localRot, angle, axis)
    def rotateLocalX(self, angle:float, radian=False):
        '''rotate around local x axis'''
        self.rotateLocalAxis(self.localRight, angle, radian)
    def rotateLocalY(self, angle:float, radian=False):
        '''rotate around local y axis'''
        self.rotateLocalAxis(self.localUp, angle, radian)
    def rotateLocalZ(self, angle:float, radian=False):
        '''rotate around local z axis'''
        self.rotateLocalAxis(self.localForward, angle, radian)
    def rotateGlobalAxis(self, globalAxis:glm.vec3, angle:float, radian=False):
        if not radian:
            angle = glm.radians(angle)
        newGlobalPos = glm.rotate(self.globalPos, angle, globalAxis)
        newGlobalForward = glm.rotate(self.forward, angle, globalAxis)
        self.globalPos = newGlobalPos
        self.forward = newGlobalForward

    @property
    def localForward(self)->glm.vec3:
        '''return vec3(0, 0, -1)'''
        return glm.vec3(0, 0, -1)
    @property
    def localUp(self)->glm.vec3:
        '''return vec3(0, 1, 0)'''
        return glm.vec3(0, 1, 0)
    @property
    def localRight(self)->glm.vec3:
        '''return vec3(1, 0, 0)'''
        return glm.vec3(1, 0, 0)

    @property
    def forward(self)->glm.vec3:
        '''return z- direction in local space transform to global space'''
        return self.transformDirection(self.localForward)
    @forward.setter
    def forward(self, value):
        '''set z- direction in local space transform to global space. Keep up direction unchanged'''
        newRight = glm.normalize(glm.cross(self.up, value))
        newUp = glm.normalize(glm.cross(value, newRight))
        self._localRot = glm.quatLookAt(value, newUp)
    @property
    def up(self)->glm.vec3:
        '''return y+ direction in local space transform to global space'''
        return self.transformDirection(self.localUp)
    @up.setter
    def up(self, value):
        '''set y+ direction in local space transform to global space. Keep forward direction unchanged'''
        newRight = glm.normalize(glm.cross(value, self.forward))
        newForward = glm.normalize(glm.cross(newRight, value))
        self._localRot = glm.quatLookAt(newForward, value)
    @property
    def right(self)->glm.vec3:
        '''return x+ direction in local space transform to global space'''
        return self.transformDirection(self.localRight)
    @right.setter
    def right(self, value):
        '''set x+ direction in local space transform to global space. Keep up direction unchanged'''
        newForward = glm.normalize(glm.cross(value, self.up))
        newUp = glm.normalize(glm.cross(newForward, value))
        self._localRot = glm.quatLookAt(newForward, newUp)
    def lookAt(self, globalTarget:Union['Transform','GameObject', glm.vec3, Sequence]):
        '''rotate transform to make forward direction point to globalTarget, and up direction point to up'''
        from runtime.gameObj import GameObject
        if isinstance(globalTarget, GameObject):
            globalTarget = globalTarget.transform.globalPos
        elif isinstance(globalTarget, Transform):
            globalTarget = globalTarget.globalPos
        target = self.inverseTransformPoint(globalTarget)
        oldForward = self.localForward
        newForward = glm.normalize(target - self.localPos)
        rotationAxis = glm.normalize(glm.cross(oldForward, newForward))
        rotationAngle = glm.acos(glm.dot(oldForward, newForward))
        self.rotateLocalAxis(rotationAxis, rotationAngle, True)
    # endregion

    # region position
    @property
    def globalPos(self)->glm.vec3:
        '''return the position of this transform in global space'''
        return self.transformPoint(self.localPos)
    @globalPos.setter
    def globalPos(self, value):
        self.setGlobalPos(value.x, value.y, value.z)
    def setGlobalPos(self, x,y,z):
        '''set the position of this transform in global space'''
        self.localPos = self.inverseTransformPoint(glm.vec3(x,y,z))
    def setGlobalPosX(self, x):
        globalPos = self.globalPos
        self.setGlobalPos(x, globalPos.y, globalPos.z)
    def setGlobalPosY(self, y):
        globalPos = self.globalPos
        self.setGlobalPos(globalPos.x, y, globalPos.z)
    def setGlobalPosZ(self, z):
        globalPos = self.globalPos
        self.setGlobalPos(globalPos.x, globalPos.y, z)
    # endregion

    # region scale
    @property
    def globalScale(self)->glm.vec3:
        '''return the scale of this transform in global space'''
        if self.parent is None:
            return self.localScale
        return (self.parent.globalScaleMatrix * glm.vec4(self.localScale, 1)).xyz
    @property
    def globalScaleX(self)->float:
        return self.globalScale.x
    @globalScaleX.setter
    def globalScaleX(self, value):
        self.setGlobalScaleX(value)
    @property
    def globalScaleY(self)->float:
        return self.globalScale.y
    @globalScaleY.setter
    def globalScaleY(self, value):
        self.setGlobalScaleY(value)
    @property
    def globalScaleZ(self)->float:
        return self.globalScale.z
    @globalScaleZ.setter
    def globalScaleZ(self, value):
        self.setGlobalScaleZ(value)
    @globalScale.setter
    def globalScale(self, value):
        self.setGlobalScale(value.x, value.y, value.z)
    def setGlobalScale(self, x,y,z):
        '''set the scale of this transform in global space'''
        scale = glm.vec3(x, y, z)
        if self.parent is None:
            self.localScale = scale
        else:
            self.localScale = (self.parent.inverseGlobalScaleMatrix * glm.vec4(scale, 1)).xyz
    def setGlobalScaleX(self, x):
        globalScale = self.globalScale
        self.setGlobalScale(x, globalScale.y, globalScale.z)
    def setGlobalScaleY(self, y):
        globalScale = self.globalScale
        self.setGlobalScale(globalScale.x, y, globalScale.z)
    def setGlobalScaleZ(self, z):
        globalScale = self.globalScale
        self.setGlobalScale(globalScale.x, globalScale.y, z)
    # endregion

    # region transform
    @property
    def translationMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local position'''
        return glm.translate(self.localPos)
    @property
    def rotationMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local rotation'''
        return glm.mat4_cast(self._localRot)
    @property
    def scaleMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local scale'''
        return glm.scale(self.localScale)
    @property
    def transformMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to local space'''
        return self.translationMatrix * self.rotationMatrix * self.scaleMatrix
    @property
    def globalTransformMatrix(self)->glm.mat4:
        '''return the matrix that can transform a vector to global space'''
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
        '''transform a point from local space to global space'''
        if self.parent is None:
            return localPos
        return (self.parent.globalTransformMatrix * glm.vec4(localPos, 1)).xyz
    def inverseTransformPoint(self, globalPos)->glm.vec3:
        '''transform a point from global space to local space'''
        if self.parent is None:
            return globalPos
        return (self.parent.inverseGlobalTransformMatrix * glm.vec4(globalPos, 1)).xyz

    def transformDirection(self, localDir)->glm.vec3:
        '''transform a direction from local space to global space (return a normalized vector)'''
        return glm.normalize(self.globalTransformMatrix * glm.vec4(localDir, 0)).xyz
    def inverseTransformDirection(self, globalDir)->glm.vec3:
        '''transform a direction from global space to local space (return a normalized vector)'''
        return glm.normalize(self.inverseGlobalTransformMatrix * glm.vec4(globalDir, 0)).xyz
    # endregion