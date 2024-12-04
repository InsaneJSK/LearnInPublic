class yrange:
    #this is both iterator and iterable, so an iter generated from this will get consumed
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            i, self.i = self.i, self.i+1
            return i
        else:
            raise StopIteration()
y = yrange(5)
print(list(y))
print(list(y))


class zrange:
    #iterable
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return zrange_iter(self.n)

class zrange_iter:
    #iterator
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()

z = zrange(10)
print(list(z))
print(list(z))