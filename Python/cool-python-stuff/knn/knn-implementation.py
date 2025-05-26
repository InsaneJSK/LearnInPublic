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

# %% [markdown]
# # KNN

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
X = pd.read_csv("xdata.csv")
y = pd.read_csv("ydata.csv")
X_data, y_data = X.values, y.values

# %%
X_data.shape

# %%
y_data[:, 1].shape

# %%
plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data[:, 1])
plt.show()


# %%
def distance_formula(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


# %%
distance_formula(X_data[0], X_data[1])


# %%
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X = X
        self.y = y
    def distance_formula(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    def predict(self, p):
        distances = []
        for i in range(len(self.X)):
            distances.append((self.distance_formula(self.X[i], p), i))
        distances = sorted(distances, key= lambda x: x[0])
        points_with_distances = distances[:self.k]
        classes = []
        for dist, i in points_with_distances:
            classes.append(self.y[i])
        class_, counts = np.unique(classes, return_counts=True)
        ix = np.argmax(counts)
        pred = {"Class": class_[ix], "prob":np.max(counts)/np.sum(counts)}
        return pred


# %%
knn = KNNClassifier(k=15)

# %%
knn.fit(X_data, y_data)

# %%
lis = []
for i in range(399):
    lis.append(knn.predict(X_data[i])["Class"])

# %%
liss = list(y_data[:, 1])

# %%
lis

# %%
val = 0
for i in range(len(lis)):
    if liss[i] == lis[i]:
        val+=1
print(val/len(lis))

# %%
