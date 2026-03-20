import math
from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    @abstractmethod
    def perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length: float, width: float):
        self.length = length
        self.width = width

    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return False
        return self.length == other.length and self.width == other.width

    def area(self) -> float:
        return self.length * self.width
    def perimeter(self) -> float:
        return 2 * (self.length + self.width)


class Square(Rectangle):
    def __init__(self, side: float):
        super().__init__(side, side)
