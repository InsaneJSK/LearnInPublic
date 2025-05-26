# ---
# jupyter:
#   jupytext:
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
import matplotlib.pyplot as plt
import numpy as np

# %%
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

a, b = np.meshgrid(a, b)
print(a)
print(b)

# %%
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(a, b, a+b, cmap='coolwarm')
plt.show()

# %%
a = np.arange(-1, 1, 0.02)
b = a
a, b = np.meshgrid(a, b)


# %%
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(a, b, a**2+b**2, cmap="rainbow")
plt.show()

# %%
fig = plt.figure()
axes = fig.add_subplot(111, projection="3d")
axes.contour(a, b, a**2+b**2, cmap="rainbow")
plt.show()
