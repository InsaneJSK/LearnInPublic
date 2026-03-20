import pytest
import source.shapes as shapes

def test_area(rect):
    result = rect.area()
    assert result == 20

def test_perimeter(rect):
    result = rect.perimeter()
    assert result == 18

def test_not_eq(rect, rect2):
    assert rect != rect2

def test_not_same_area(rect, rect2):
    assert rect.area() != rect2.area()
