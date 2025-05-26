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
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# %%
cpu_cnt = multiprocessing.cpu_count()
print(cpu_cnt)

# %%
params = [
    {
    'kernel':['linear','rbf','poly','sigmoid'],
    'C' : [0.1,0.2,0.4,0.5,1.0,2.0,5.0]
    }
]

# %%
svc = svm.SVC()
gs = GridSearchCV(estimator=svc,param_grid = params,scoring='accuracy',cv=5,n_jobs=cpu_cnt)

# %%
from sklearn.datasets import load_digits

# %%
digits = load_digits()

# %%
X = digits.data
Y  = digits.target
print(X.shape)
print(Y.shape)

# %%
from sklearn.model_selection import cross_val_score

# %%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cross_val_score(lr,X,Y,scoring="accuracy",cv=5).mean()

# %%
svc = svm.SVC()
cross_val_score(svc,X,Y,scoring="accuracy",cv=5).mean()

# %%

gs.fit(X,Y)

# %%

gs.best_estimator_

# %%
gs.best_score_
