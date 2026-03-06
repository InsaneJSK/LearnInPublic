def add(n1: int, n2: int) -> int:
    return n1 + n2

def divide(n1: int, n2: int) -> float:
    if n2 == 0:
        raise ValueError
    return n1 / n2
