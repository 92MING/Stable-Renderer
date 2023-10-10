'''GameObject is the class contains components and behaviors.'''

import numpy as np
from abc import abstractmethod
from static.data_types.vector import Vector
from static.data_types.matrix import Matrix
from utils.base_clses import NamedObj

from enum import Enum


class GameObject(NamedObj):
    game_object_tag_search_map = {}  # Map from tag to GameObject
    root_game_objects = []

    def __init__(self, name="", active=True, parent=None):
        super().__init__(name)

        self.active = active
        self.name = name
        self.tags = set()
        self.parent = parent
        self.children = []
        self.attributes = {}
        self._transform = None

        if parent is None:
            GameObject.root_game_objects.append(self)
        else:
            self.parent.children.append(self)

    def is_active(self):
        return self.active

    def set_active(self, set_val):
        if set_val == self.active:
            return
        for child in self.children:
            child.set_active(set_val)
        self.active = set_val
        for att_list in self.attributes.values():
            for att in att_list:
                if att.is_enable():
                    if not set_val:
                        att.on_disable()
                    else:
                        att.on_enable()

    def get_name(self):
        return self.name

    # def has_tag(self, tag):
    #     return tag in self.tags

    # def add_tag(self, tag):
    #     if self.has_tag(tag):
    #         return
    #     self.tags.add(tag)
    #     if tag not in GameObject.game_object_tag_search_map:
    #         GameObject.game_object_tag_search_map[tag] = set()
    #     GameObject.game_object_tag_search_map[tag].add(self)

    # def remove_tag(self, tag):
    #     if not self.has_tag(tag):
    #         return
    #     self.tags.remove(tag)
    #     GameObject.game_object_tag_search_map[tag].remove(self)

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

    def set_parent(self, new_parent):
        """
        Set parent object. If `new_parent` is None, this object will be moved to root object list.
        """
        if new_parent == self.parent:
            return
        if self.parent is not None:
            self.parent.children.remove(self)
        else:
            GameObject.root_game_objects.remove(self)
        self.parent = new_parent
        if self.parent is not None:
            self.parent.children.append(self)
        else:
            GameObject.root_game_objects.append(self)

    def move_to_root_object_list(self):
        self.set_parent(None)

    def get_parent(self):
        return self.parent

    def has_child(self, child):
        return child in self.children

    def get_all_children(self):
        return self.children

    def get_child(self, index):
        return self.children[index]

    def child_count(self):
        return len(self.children)

    def change_sibling_index(self, index):
        if index >= len(self.children):
            return
        if self.parent is None:
            if index >= len(GameObject.root_game_objects):
                return
            GameObject.root_game_objects.remove(self)
            GameObject.root_game_objects.insert(index, self)
        else:
            if index >= len(self.parent.children):
                return
            self.parent.children.remove(self)
            self.parent.children.insert(index, self)

    def get_sibling_index(self):
        if self.parent is None:
            return GameObject.root_game_objects.index(self) if self in GameObject.root_game_objects else -1
        else:
            return self.parent.children.index(self) if self in self.parent.children else -1

    def has_attribute_type(self, att_type):
        for att_list in self.attributes.values():
            for att in att_list:
                if isinstance(att, att_type):
                    return True
        return False

    def has_attribute(self, att):
        for att_list in self.attributes.values():
            if att in att_list:
                return True
        return False

    def get_first_attribute_type(self, att_type):
        for att_list in self.attributes.values():
            for att in att_list:
                if isinstance(att, att_type):
                    return att
        return None

    def get_all_attribute_type(self, att_type):
        result = []
        for att_list in self.attributes.values():
            for att in att_list:
                if isinstance(att, att_type):
                    result.append(att)
        return result

    def remove_attribute_type(self, att_type):
        to_remove = []
        for key, att_list in self.attributes.items():
            for att in att_list:
                if isinstance(att, att_type):
                    to_remove.append(att)
            for att in to_remove:
                att_list.remove(att)
            if not att_list:
                del self.attributes[key]

    def remove_attribute(self, att):
        for key, att_list in self.attributes.items():
            if att in att_list:
                att_list.remove(att)
            if not att_list:
                del self.attributes[key]

    def get_all_attributes(self):
        return self.attributes

    def add_attribute(self, att_class, *args, **kwargs):
        if att_class == TransformAttribute:
            return self.transform()
        new_att = att_class(self, *args, **kwargs)
        priority = new_att.get_attribute_type_priority()
        if priority not in self.attributes:
            self.attributes[priority] = []
        self.attributes[priority].append(new_att)
        return new_att

    def set_all_attribute_status(self, set_val):
        for att_list in self.attributes.values():
            for att in att_list:
                att.set_enable(set_val)

    def transform(self) -> 'TransformAttribute':
        if self._transform is None:
            self._transform = TransformAttribute(self, True)
        return self._transform

    @staticmethod
    def find_game_objects_with_name(name):
        return GameObject.game_object_name_search_map.get(name, set())

    @staticmethod
    def find_game_objects_with_tag(tag):
        return GameObject.game_object_tag_search_map.get(tag, set())

    @staticmethod
    def get_root_game_objects():
        return GameObject.root_game_objects

    @staticmethod
    def clear_all_game_object():
        for obj in list(GameObject.root_game_objects):
            obj.destroy()

    @staticmethod
    def run_game_object(obj, stage):
        if obj.is_active():
            for att_list in obj.attributes.values():
                for att in att_list:
                    if att.is_enable():
                        att.run_attribute(stage)
            for child in obj.children:
                GameObject.run_game_object(child, stage)  # Depth-first


class EngineRunStage(Enum):
    FIXED_UPDATE = 1
    UPDATE = 2
    LATE_UPDATE = 3


class GameAttributeBasic:
    def __init__(self, game_obj: GameObject, enable=True):
        self.enable = enable
        self.started = False
        self.game_object = game_obj
        self.awake()
        if enable:
            self.on_enable()
        else:
            self.on_disable()

    @abstractmethod
    def get_attribute_type_priority(self):
        raise NotImplementedError

    @abstractmethod
    def get_attribute_type_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_attribute_type(self):
        raise NotImplementedError

    @abstractmethod
    def destroy(self):
        raise NotImplementedError

    # Default implementations for the lifecycle methods
    @abstractmethod
    def awake(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def fixed_update(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def late_update(self):
        pass

    @abstractmethod
    def on_destroy(self):
        pass

    @abstractmethod
    def on_enable(self):
        pass

    @abstractmethod
    def on_disable(self):
        pass

    def set_enable(self, value):
        if value != self.enable:
            self.enable = value
            if value:
                self.on_enable()
            else:
                self.on_disable()

    def is_enable(self):
        return self.enable

    def get_game_object(self):
        return self.game_object

    def equal_to_attribute_type(self, att_type):
        return att_type == self.get_attribute_type()

    def run_attribute(self, stage):
        if not self.started:
            self.started = True
            self.start()

        if self.enable:
            if stage == EngineRunStage.FIXED_UPDATE:
                self.fixed_update()
            elif stage == EngineRunStage.UPDATE:
                self.update()
            elif stage == EngineRunStage.LATE_UPDATE:
                self.late_update()


class GameAttribute(GameAttributeBasic):
    all_att_instance = set()

    def __init__(self, game_obj, enable, attribute_type_name, att_type_priority=0):
        super().__init__(game_obj, enable)
        self._attribute_type_name = attribute_type_name
        self._att_type_priority = att_type_priority
        GameAttribute.all_att_instance.add(self)

    def destroy(self):
        GameAttribute.all_att_instance.remove(self)
        del self

    def get_attribute_type(self):
        return type(self)

    def get_attribute_type_priority(self):
        return self._att_type_priority

    def get_attribute_type_name(self):
        return self._attribute_type_name

    @staticmethod
    def get_all_att_instance(cls):
        return cls.all_att_instance


class TransformAttribute(GameAttribute):
    transform_attribute_type_name = "TransformAttribute"

    def __init__(self, game_obj, enable=True):
        super().__init__(game_obj, enable, TransformAttribute.transform_attribute_type_name)
        self.rotation = Vector(0.0, 0.0, 0.0)
        self.position = Vector(0.0, 0.0, 0.0)
        self.scale = Vector(1.0, 1.0, 1.0)

    def set_enable(self, enable):
        print("TransformAttribute can't change enable status.")

    def get_parent_transform(self):
        par = self.game_object.get_parent()
        if par is None:
            return None
        return par.transform()

    def tidy_up_rotation(self):
        self.rotation %= 360.0

    def get_local_rotation(self):
        self.tidy_up_rotation()
        return self.rotation

    def set_local_rotation(self, x=None, y=None, z=None):
        if x is not None:
            self.rotation[0] += x
        if y is not None:
            self.rotation[1] += y
        if z is not None:
            self.rotation[2] += z
        self.tidy_up_rotation()

    def get_world_pos(self):
        parent_transform = self.get_parent_transform()
        if parent_transform is None:
            return self.position
        return parent_transform.get_world_pos() + self.position

    def get_world_rotation(self):
        """
        Return the world rotation in degree of this transform.
        """
        par_transform = self.get_parent_transform()
        if par_transform is None:
            return self.rotation
        rot = par_transform.get_world_rotation() + self.rotation
        rot %= 360
        return rot

    def get_world_scale(self):
        par_transform = self.get_parent_transform()
        if par_transform is None:
            return self.scale
        return par_transform.get_world_scale() * self.scale

    def transform_dir(self, dir: Vector):
        real_rotation = self.get_world_rotation()
        rotation_matrix = Matrix.Rotation(*real_rotation, radian=False)
        return (rotation_matrix * Vector(*dir, 1.0)).xyz

    def transform_point(self, point):
        real_pos = self.get_world_pos()
        return real_pos + point

    def inverse_transform_point(self, point):
        real_pos = self.get_world_pos()
        return point - real_pos

    def inverse_transform_dir(self, dir):
        real_rotation = self.get_world_rotation()
        rotation_matrix = Matrix.Rotation(*real_rotation, radian=False)
        return (rotation_matrix.I * Vector(*dir, 1.0)).xyz

    def set_world_pos(self, target_pos):
        par_transform = self.get_parent_transform()
        if par_transform is None:
            self.position = target_pos
        else:
            real_pos = par_transform.get_world_pos()
            self.position = target_pos - real_pos

    def set_world_rotation(self, target_rotation):
        par_transform = self.get_parent_transform()
        if par_transform is None:
            self.rotation = target_rotation
        else:
            real_rotation = par_transform.get_world_rotation()
            self.rotation = target_rotation - real_rotation
            self.tidy_up_rotation()

    def set_world_scale(self, target_scale):
        par_transform = self.get_parent_transform()
        if par_transform is None:
            self.scale = target_scale
        else:
            real_scale = par_transform.get_world_scale()
            self.scale = target_scale / real_scale

    # TODO: Correct this function
    def set_forward(self, forward_dir: Vector):
        real_rotation = self.get_world_rotation()
        rotation_matrix = Matrix.Rotation(*real_rotation, radian=False)
        right = (rotation_matrix * Vector(1.0, 0.0, 0.0, 1.0)).xyz
        up = (rotation_matrix * Vector(0.0, 1.0, 0.0, 1.0)).xyz
        forward = (rotation_matrix * Vector(0.0, 0.0, 1.0, 1.0)).xyz

        new_forward = forward_dir.normalize
        new_right = (up.cross(new_forward)).normalize
        new_up = (new_forward.cross(new_right)).normalize
        new_rotation = Vector(
            np.degrees(np.arcsin(new_up.z)),
            np.degrees(np.arctan(new_up.x / new_up.y)),
            np.degrees(np.arctan(new_right.z / new_forward.z))
        )
        self.set_world_rotation(new_rotation)

    def look_at(self, position):
        real_pos = self.get_world_pos()
        forward = position - real_pos
        self.set_forward(forward)

    def transform_obj(self, transform):
        transform.set_world_pos(self.transform_point(transform.get_world_pos()))
        transform.set_world_rotation(self.get_world_rotation() + transform.get_world_rotation())

    def forward(self):
        return self.transform_dir(Vector(0.0, 0.0, 1.0))

    # TODO: Correct this function
    def rotate(self, rotate_axis, rotate_angle):
        rotation_matrix = mat4(1.0)
        rotation_matrix = rotate(rotation_matrix, radians(rotate_angle), rotate_axis)
        self.position = rotation_matrix * vec4(self.position, 1.0)
        self.set_forward(rotation_matrix * vec4(self.forward(), 1.0))

    def backward(self):
        return -self.forward()

    def right(self):
        return self.transform_dir(Vector(1.0, 0.0, 0.0))

    def set_right(self, right_dir: Vector):
        new_forward = (right_dir.cross(self.right())).normalize
        self.set_forward(new_forward)

    def left(self):
        return -self.right()

    def up(self):
        return self.transform_dir(Vector(0.0, 1.0, 0.0))

    def set_up(self, up_dir: Vector):
        new_right = (up_dir.cross(self.up())).normalize
        self.set_right(new_right)

    def down(self):
        return -self.up()

    def get_model_matrix(self):
        real_pos = self.get_world_pos()
        real_rotation = self.get_world_rotation()
        real_scale = self.get_world_scale()
        model_matrix *= Matrix.Translation(*real_pos)
        model_matrix *= Matrix.Rotation(*real_rotation, radian=False)
        model_matrix *= Matrix.Scale(*real_scale)
        return model_matrix

    # Continue here...
