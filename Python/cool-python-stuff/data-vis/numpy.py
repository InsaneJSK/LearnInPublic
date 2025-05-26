# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Numpy

# %%
import numpy as np

# %%
lis = [[1, 2], [3, 4]]
a = np.array(lis, dtype = 'int64')
print(a)
print(type(a))
print(a.dtype)
print(a.shape)
print(a.T)

# %%
b = np.zeros((2, 2))
c = np.ones((2, 2))
d = np.full((2, 2), 5)
e = np.eye(2)
f = np.random.random((2, 2))
print(b)
print(c)
print(d)
print(e)
print(f)

# %%
a+b

# %%
np.add(a, b)

# %%
a-b

# %%
np.subtract(a, b)

# %%
a*b

# %%
np.multiply(a, b)

# %%
a/f

# %%
np.divide(a, f)

# %%
np.sqrt(a)

# %%
a.dot(c)

# %%
np.dot(a, b)

# %%
sum(a)

# %%
np.sum(a, axis = 0)

# %%
np.sum(a, axis = 1)

# %%
np.stack((a, b), axis = 0)

# %%
np.stack((a, b), axis = 1)

# %%
a.reshape(1, 4)

# %%
print(a)

# %%
np.random.shuffle(a)

# %%
print(np.arange(20).reshape(5, -1))

# %%
np.random.choice(a.reshape(-1, ))

# %%
