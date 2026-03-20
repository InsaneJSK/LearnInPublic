import pytest
import source.shapes as shapes

@pytest.mark.parametrize("side, expected_area", [
    (2, 4), (3, 9), (4, 16), (5, 25)])
def test_square_area(side, expected_area):
    square = shapes.Square(side)
    assert square.area() == expected_area

@pytest.mark.parametrize("side, expected_perimeter", [
    (2, 8), (3, 12), (4, 16), (5, 20)])
def test_square_perimeter(side, expected_perimeter):
    square = shapes.Square(side)
    assert square.perimeter() == expected_perimeter
