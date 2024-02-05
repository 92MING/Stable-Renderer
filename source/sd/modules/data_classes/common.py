from dataclasses import dataclass
from typing import Tuple

@dataclass
class Rectangle:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]