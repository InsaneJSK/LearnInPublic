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
from sklearn.naive_bayes import GaussianNB

# %%
Gnb = GaussianNB()

# %%
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# %%
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=4)

# %%

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%
print(X[0])
print(X.shape)

# %%
Gnb.fit(X, y)

# %%
Gnb.score(X, y)

# %%
ypred = Gnb.predict(X)
print(ypred)
