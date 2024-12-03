"""
Peter wants to generate some prime numbers for his cryptosystem. Help him! Your task is to generate all prime numbers between two given numbers!


Input Format
The input begins with the number t of test cases in a single line (t<=10). In each of the next t lines, there are two numbers m and n separated by a space.

Output Format
For every test case print all prime numbers p such that m <= p <= n, all primes per line, test cases separated by an empty line.
"""

#Sieve of Eratosthenes
"""
x = [True for i in range(0, 101)]
primes = []
for i in range(2, 101):
    if x[i] == True:
        for j in range(i**2, 101, i):
            x[j] = False
for i in range(2, 101):
    if x[i] == True:
        primes.append(i)
"""

t = int(input())
lis, lisb = [], []
for i in range(t):
    a, b = map(int, input().split())
    lis.append([a, b])
    for a, b in lis:
        lisb.append(b)
maxb = max(lisb)

x = [True for i in range(maxb+1)]
for i in range(2, maxb+1):
    if x[i] == True:
        for j in range(i**2, maxb+1, i):
            x[j] = False

primes = [i for i in range(2, maxb + 1) if x[i]]

for (a, b) in lis:
    for i in primes:
        if i < a:
            continue
        elif i >= a and i <= b:
            print(i, end = " ")
        elif i > b:
            break
    print()
