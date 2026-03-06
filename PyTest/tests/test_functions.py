import pytest
import source.functions as functions

def test_add():
    result = functions.add(2, 3)
    assert result == 5

def test_divide():
    result = functions.divide(10, 2)
    assert result == 5.0

def test_divide_by_zero():
    with pytest.raises(ValueError):
        functions.divide(10, 0)
