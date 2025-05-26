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
X = pd.read_csv("Linear_X_Train.csv")
Y = pd.read_csv("Linear_Y_Train.csv")
X = X.values
Y = Y.values

X = (X- X.mean())/X.std()

# %%
plt.scatter(X, Y)
plt.show()

# %%
X.shape

# %%
Y.shape


# %%
def hypothesis(x, theta):
    y_ = theta[0] + theta[1]*x
    return y_


# %%
def gradient(X, Y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        y_ = hypothesis(x, theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y)*x
    return grad/m


# %%
def error(X, Y, theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i], theta)
        total_error += (y_ - Y[i])**2
    return total_error/m


# %%
def gradientDescent(X, Y, max_steps = 100, learning_rate=0.1):
    theta = np.zeros((2,))
    error_list = []
    theta_list = []
    for i in range(max_steps):
        grad = gradient(X, Y, theta)
        e = error(X, Y, theta)
        error_list.append(e)
        theta[0] -= learning_rate*grad[0]
        theta[1] -= learning_rate*grad[1]
        theta_list.append((theta[0],theta[1]))
    return theta, error_list, theta_list


# %%
theta, error_list, theta_list = gradientDescent(X, Y)

# %%
theta

# %%
error_list

# %%
plt.plot(error_list)

# %%
y_ = hypothesis(X, theta)
print(y_)

# %%
plt.scatter(X, Y)
plt.plot(X, y_, color="orange", label = "predictions")
plt.legend()
plt.show()

# %%
X_test = pd.read_csv("Linear_X_Test.csv")
y_test = hypothesis(X_test, theta)
print(y_test)

# %%
df = pd.DataFrame(y_test)
df.columns = ["y"]
df

# %%
df.to_csv('y_predictions.csv', index=False)


# %%
def r2_score(Y, Y_):
    num = np.sum((Y-Y_)**2)
    denom = np.sum((Y-Y.mean())**2)
    score = (1 - num/denom)
    return score*100


# %%
r2_score(Y, y_)

# %%
theta

# %%
T0 = np.arange(-40,40,1)
T1 = np.arange(40,120,1)

T0, T1 = np.meshgrid(T0, T1)
J = np.zeros(T0.shape)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_ = T1[i,j]*X + T0[i, j]
        J[i,j] = np.sum((Y-y_)**2)/Y.shape[0]

# %%
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(T0,T1,J,cmap='rainbow')
plt.show()

# %%
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.contour(T0,T1,J,cmap='rainbow')
plt.show()

# %%
theta_list = np.array(theta_list)
theta_list.shape

# %%
plt.plot(theta_list[:, 0], label="Theta 0")
plt.plot(theta_list[:, 1], label="Theta 1")
plt.legend()

# %%
#Trajectory traced by theta updates in the loss function
error_list = np.array(error_list).reshape((100,))
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(T0,T1,J,cmap='rainbow')
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list)
plt.show()

# %%
#Trajectory traced by theta updates in the loss function
error_list = np.array(error_list).reshape((100,))
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.contour(T0,T1,J,cmap='rainbow')
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list)
plt.show()

# %%
plt.contour(T0, T1, J, cmap = "coolwarm")
plt.scatter(theta_list[:, 0], theta_list[:, 1])
plt.show()

# %%
np.save("Theta_list.npy", theta_list)
