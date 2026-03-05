#Method Resolution Order

class A:
    X = "A"
class B(A):
    X = "B"
class C(A):
    X = "C"
class D(C):
    X = "D"
class E(B, D):
    X = "E"

print(E.__mro__)
