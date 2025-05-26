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
import pandas as pd
from sklearn.datasets import make_regression

# %%
X, Y = make_regression(n_samples=400, n_features=1, n_informative=1, noise=1.8, random_state=11)
Y = Y.reshape((-1,1))
print(X.shape)
print(Y.shape)

# %%
#Normalize
X = (X-X.mean())/X.std()

plt.figure()
plt.scatter(X, Y)
plt.title("Normalization Data")
plt.show()

# %%
ones = np.ones((X.shape[0], 1))
X_ = np.hstack((X, ones))
print(X_.shape)
print(X_[:5, :])


# %%
def predict(X, theta):
    return np.dot(X, theta)

def getThetaCloseForm(X, Y):
    Y = np.mat(Y)
    return np.linalg.pinv(np.dot(X.T, X))*np.dot(X.T, Y)


# %%
theta = getThetaCloseForm(X_, Y)
print(theta)

# %%
plt.figure()
plt.scatter(X, Y)
plt.plot(X, predict(X_, theta), color = "red")
plt.title("Prediction")
plt.show()
