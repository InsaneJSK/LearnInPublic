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
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# %%
#Dataset Preparation
digits = load_digits()

# %%
X = digits.data
y = digits.target

# %%
plt.imshow(X[1].reshape((8, 8)), cmap = "grey")
print(y[1])
plt.show()

# %%
#Train models

# %%
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# %%
mnb = MultinomialNB()
gnb = GaussianNB()

# %%
mnb.fit(X, y)
gnb.fit(X, y)

# %%
print(mnb.score(X, y))
print(gnb.score(X, y))

# %%
cross_val_score(gnb, X, y, scoring="accuracy", cv=10).mean()

# %%
cross_val_score(mnb, X, y, scoring="accuracy", cv=10).mean()
