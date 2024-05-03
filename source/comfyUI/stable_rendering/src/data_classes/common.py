from typing import Tuple
from dataclasses import dataclass


@dataclass
class Rectangle:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    def is_in_rectangle(self, coordinates: Tuple[int, int]):
        return (coordinates[0] < self.bottom_right[0] and coordinates[0] > self.top_left[0]) and \
                    (coordinates[1] < self.bottom_right[1] and coordinates[1] > self.top_left[1])


