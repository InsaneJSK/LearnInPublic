# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
X = np.arange(10)
Y = (X-5)**2
print(X, Y)

# %%
#Given a function f(x) we want to find the value of x that minimizes f
#Visualization
plt.plot(X, y)
plt.ylabel("f(x)")
plt.xlabel("x")

# %%
x = 0
lr = 0.1
error = []

#50 steps in the downward direction
plt.plot(X, Y)
for i in range(50):
    grad = 2*(x-5)
    x = x - lr*grad
    y = (x-5)**2
    error.append(y)
    plt.scatter(x, y)
    print(x)

# %%
plt.plot(error)
plt.show()
