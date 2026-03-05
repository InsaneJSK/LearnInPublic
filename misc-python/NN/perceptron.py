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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# %%
X, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=11)
print(X.shape, y.shape)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
plt.show()


# %%
def sigmoid(z):
    return (1.0)/(1+np.exp(-z))


# %%
def predict(X, weights):
    """X -> mx(n+1) matrix, w --> (nX1,) vector"""
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions

def loss(X, y, weights):
    """binary cross entropy"""
    y_ = predict(X, weights)
    cost = np.mean(-y*np.log(y_) - (1-y)*np.log(1 - y_))
    return cost

def update(X, y, weights, learning_rate):
    """Perform weights updates from 1 epoch"""
    y_ = predict(X, weights)
    dw = np.dot(X.T, y_ - y)
    m = X.shape[0]
    weights = weights - learning_rate*dw/(float(m))
    return weights

def train(X, y, learning_rate=0.5, maxEpochs=100):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    weights = np.zeros(X.shape[1])
    for epoch in range(maxEpochs):
        weights = update(X, y, weights, learning_rate)
        if epoch%10==0:
            l = loss(X, y, weights)
            print("Epoch %d loss %.4f"%(epoch, l))
    
    return weights


# %%
weights = train(X, y, learning_rate=0.8, maxEpochs=1000)


# %%
def getPredictions(X_Test, weights, labels=True):
    if X_Test.shape[0] != weights.shape[0]:
        ones = np.ones((X_Test.shape[0], 1))
        X_Test = np.hstack((ones, X_Test))
    probs = predict(X_Test, weights)
    if not labels:
        return probs
    else:
        labels = np.zeros(probs.shape)
        labels[probs>=0.5] = 1
        return labels


# %%
x1 = np.linspace(-8, 2, 10)
x2 = -(weights[0] + weights[1]*x1)/weights[2]

# %%
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Accent)
plt.plot(x1, x2, c="red")
plt.show()

# %%
Y_ = getPredictions(X, weights, labels=True)
training_acc = np.sum(Y_ == y)/y.shape[0]
print(training_acc)
