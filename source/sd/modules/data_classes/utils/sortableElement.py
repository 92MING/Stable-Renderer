from typing import Any, Protocol
from abc import abstractmethod

class Comparable(Protocol):
    @abstractmethod
    def __lt__(self, other)->bool:
        pass

class SortableElement(Comparable):
    def __init__(self, value: Comparable, object: Any):
        self._value = value
        self._object = object

    @property
    def Value(self):
        return self._value

    @property
    def Object(self):
        return self._object

    def __eq__(self, other):
        if isinstance(other, SortableElement):
            return self.Value == other.Value
        return False

    def __lt__(self, other):
        if isinstance(other, SortableElement):
            return self.Value < other.Value
        raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, SortableElement):
            return self.Value > other.Value
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self._value})"

    def __str__(self):
        return f"{self.__repr__}, object: {self._object}"
    
__all__ = ['SortableElement']