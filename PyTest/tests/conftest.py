import pytest
import source.shapes as shapes

@pytest.fixture
def rect():
    return shapes.Rectangle(4, 5)

@pytest.fixture
def rect2():
    return shapes.Rectangle(5, 5)