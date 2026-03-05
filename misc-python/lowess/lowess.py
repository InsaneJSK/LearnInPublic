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
# %matplotlib inline

# %%
dfx = pd.read_csv("weightedX.csv")
dfy = pd.read_csv("weightedY.csv")

# %%
X = dfx.values
Y = dfy.values

# %%
print(X.shape)
print(Y.shape)

# %%
u = X.mean()
std = X.std()
X = (X-u)/std
plt.title("Normalized Data")
plt.scatter(X, Y)
plt.show()


# %%
def getW(query_point, X, tau):
    M = X.shape[0]
    W = np.mat(np.eye(M))
    x = query_point
    for i in range(M):
        xi = X[i]
        W[i, i] = np.exp(-np.linalg.norm(xi - x) ** 2 / (2 * tau * tau))
    return W


# %%
X = np.mat(X)
Y = np.mat(Y)
M = X.shape[0]
W = getW(-1, X, 0.5)
print(W.shape)
print(W)


# %%
def predict(X, Y, query_x, tau):
    ones = np.ones((M, 1))
    X_ = np.hstack((X, ones))
    qx = np.mat([query_x, 1])
    W = getW(qx, X_, tau)
    theta = np.linalg.pinv(X_.T*(W*X_))*(X_.T*(W*Y))
    pred = np.dot(qx, theta)
    return theta, pred


# %%
theta, pred = predict(X, Y, 1.0, 1.0)

# %%
print(pred)


# %%
def plotPrediction(tau):
    X_test = np.linspace(-2, 2, 20)
    Y_test = []
    for xq in X_test:
        _, pred = predict(X, Y, xq, tau)
        Y_test.append(pred[0][0])
    Y_test = np.array(Y_test)
    XO = np.array(X)
    YO = np.array(Y)
    plt.scatter(XO, YO)
    plt.scatter(X_test, Y_test, color="red")
    plt.title(tau)
    plt.show()


# %%
t = 0.01
while t < 100:
    plotPrediction(t)
    t*=2
