import pytest
import source.shapes as shapes

class TestCircle:
    def setup_method(self, method):
        print(f"Setting up for {method.__name__}")
        self.circle = shapes.Circle(radius=5)

    def teardown_method(self, method):
        print(f"Tearing down after {method.__name__}")
        del self.circle

    def test_area(self):
        result = self.circle.area()
        assert result == pytest.approx(78.53981633974483)

    def test_perimeter(self):
        result = self.circle.perimeter()
        assert result == pytest.approx(31.41592653589793)
    