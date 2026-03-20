import pytest
import source.shapes as shapes
import time

@pytest.mark.slow
def test_area(rect):
    result = rect.area()
    assert result == 20

@pytest.mark.skip(reason="This test is skipped because it's not relevant for the current testing phase.")
def test_perimeter(rect):
    time.sleep(2)
    result = rect.perimeter()
    assert result == 18

@pytest.mark.xfail(reason="This test is expected to fail because the addition is not implemented.")
def test_not_eq(rect, rect2):
    assert rect + rect2
