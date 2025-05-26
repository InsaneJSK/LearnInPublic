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
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %%
X,Y = make_circles(n_samples=500,noise=0.02)

# %%
print(X.shape,Y.shape)

# %%
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# %%
def phi(X):
    """"Non Linear Transformation"""
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X1**2 + X2**2
    
    X_ = np.zeros((X.shape[0],3))
    print(X_.shape)
    
    X_[:,:-1] = X
    X_[:,-1] = X3

    return X_


# %%
X_ = phi(X)

# %%
print(X[:3,:])

# %%
print(X_[:3,:])


# %%
def plot3d(X,show=True):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    
    if(show==True):
        plt.show()
    return ax


# %%
ax = plot3d(X_)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# %%
lr = LogisticRegression()

# %%
acc = cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc*100))

# %%
acc = cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(3D) is %.4f"%(acc*100))

# %%
lr.fit(X_,Y)

# %%
wts = lr.coef_
print(wts)

# %%
bias = lr.intercept_

# %%
xx,yy = np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print(yy)

# %%
z = -(wts[0,0]*xx + wts[0,1]*yy+bias)/wts[0,2]
print(z)

# %%
ax = plot3d(X_,False)
ax.plot_surface(xx,yy,z,alpha=0.2)
plt.show()

# %%
from sklearn import svm

# %%
svc = svm.SVC(kernel="linear")

# %%
svc.fit(X,Y)

# %%
svc.score(X,Y)

# %%
svc = svm.SVC(kernel="rbf")
svc.fit(X,Y)
svc.score(X,Y)

# %%
svc = svm.SVC(kernel="poly")
svc.fit(X,Y)
svc.score(X,Y)


# %%
def custom_kernel(x1,x2):
    return np.square(np.dot(x1,x2.T))

svc = svm.SVC(kernel=custom_kernel)
svc.fit(X,Y)
svc.score(X,Y)
