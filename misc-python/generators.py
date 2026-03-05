#Sample Generator Function
def fib():
    prev, curr = 0, 1
    while True:
        yield curr
        prev, curr = curr, prev + curr

gen = fib()

for _ in range(10):
    print(next(gen))


#But there also exists a generator expression
gen = (x**2 for x in range(1, 11)) #Obviously nothing like tuple comprehension exists, duh
for _ in range(5):
    print(next(gen))