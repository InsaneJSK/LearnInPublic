#with

class A:
    def __init__(self, n):
        self.n = n
    def __str__(self):
        return str(self.n)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        for arg in args:
            print(arg)
        return True
    

with A(5) as a:
    print(a)
    raise 10/0
print("hello")
