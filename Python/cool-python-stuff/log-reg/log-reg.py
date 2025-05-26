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

# %%
# Data - Generate using numpy
mean_01 = np.array([1, 0.5])
cov_01 = np.array([[1, 0.1], [0.1, 1.2]])

mean_02 = np.array([4, 5])
cov_02 = np.array([[1.2, 0.1], [0.1, 1.3]])

dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)

# %%
dist_01.shape

# %%
plt.scatter(dist_01[:, 0], dist_01[:, 1], c="red", label="Class 0")
plt.scatter(dist_02[:, 0], dist_02[:, 1], c="blue", label="Class 1")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend()
plt.show()

# %%
data = np.zeros((1000, 3))

# %%
data[:500, :2] = dist_01
data[500:, :2] = dist_02

data[500:, -1] = 1.0

# %%
#Randomly shuffle the data
np.random.shuffle(data)
print(data[:10])

# %%
#Divide the data into train and test part
split = int(0.8*data.shape[0])
X_train = data[:split, :-1]
X_test = data[split:, :-1]

Y_train = data[:split, -1]
Y_test = data[split:, -1]

# %%
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.show()

# %%
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)

X_train = (X_train-x_mean)/x_std

X_test = (X_test-x_mean)/x_std

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.show()


# %%
#Logistic regression implementation
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def hypothesis(X, theta):
    """
    X = entire array (m, n+1)
    theta = np.array(n+1, 1)
    """
    return sigmoid(np.dot(X, theta))
def error(X, y, theta):
    """
    params:
    X = (m, n+1)
    y = (m, 1)
    theta = (n+1, 1)

    return:
        scale_value = loss
    """
    hi = hypothesis(X, theta)
    epsilon = 1e-15
    hi = np.clip(hi, epsilon, 1 - epsilon)  # Prevent log(0)
    e = -1*np.mean((y*np.log(hi) + ((1-y)*np.log(1-hi))))
    return e


# %%
def gradient(X, y, theta):
    """
    params:
    X = (m, n+1)
    y = (m, 1)
    theta = (n+1, 1)
    
    return:
        gradient_vector = (n+1, 1)
    """
    hi = hypothesis(X, theta)
    grad = np.dot(X.T, (y-hi))
    m = X.shape[0]
    return grad/m

def gradient_descent(X, y, lr = 0.1, max_itr=500):
    
    n = X.shape[1]
    theta = np.zeros((n, 1))

    error_list = []

    for i in range(max_itr):
        err = error(X, y, theta)
        error_list.append(err)

        grad = gradient(X, y, theta)

        theta = theta + lr*grad
    return (theta, error_list)


# %%
ones = np.ones((X_train.shape[0], 1))
X_new_train = np.hstack((ones, X_train))
print(X_new_train.shape)
print(X_new_train)
Y_train = Y_train.reshape((-1, 1))

# %%
theta, error_list = gradient_descent(X_new_train, Y_train)

# %%
plt.plot(error_list)

# %%
#Visualize Decision surface
x1 = np.arange(-3, 4)
x2 = -(theta[0] + theta[1]*x1)/theta[2]
plt.scatter(X_train[:, 0], X_train[:, 1], c = Y_train.reshape((-1,)))
plt.plot(x1, x2)
plt.show()

# %%
#Predictions and accuracy
X_new_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
print(X_new_test.shape)
print(X_new_test[:3, :])


# %%
def predict(X, theta):
    h = hypothesis(X, theta)
    output = np.zeros(h.shape)
    output[h>=0.5] = 1
    output = output.astype('int')
    return output

XT_preds = predict(X_new_train, theta)
Xt_preds = predict(X_new_test, theta)


# %%
def accuracy(actual, preds):
    actual = actual.astype('int')
    actual = actual.reshape((-1, 1))
    acc = np.sum(actual==preds)/actual.shape[0]
    return acc*100


# %%
train_Acc = accuracy(Y_train, XT_preds)
test_acc = accuracy(Y_test, Xt_preds)
print(train_Acc, test_acc)
