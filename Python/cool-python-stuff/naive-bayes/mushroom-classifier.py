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

# %%
df = pd.read_csv("mushrooms.csv")
df.head(10)

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %%
le = LabelEncoder()
ds = df.apply(le.fit_transform)

# %%
data = ds.values

# %%
X, y = data[:, 1:], data[:, 0]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
a = np.array([0, 5, 5, 1, 1, 1, 0, 1])


# %%
def prior_prob(y_train, label):
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train == label)

    return (class_examples)/float(total_examples)


# %%
y = np.array([0, 5, 5, 1, 1, 1, 1, 0, 0, 0])
prior_prob(y, 0)


# %%
def cond_prob(x_train, y_train, feature_col, feature_val, label):
    x_filtered = x_train[y_train==label]
    numerator = np.sum(x_filtered[:, feature_col]==feature_val)
    denominator = np.sum(y_train == label)
    return numerator/float(denominator)


# %%
def predict(x_train, y_train, Xtest):
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    post_probs = []
    for label in classes:
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train, y_train, f, Xtest[f], label)
            likelihood*=cond
        prior = prior_prob(y_train, label)
        post = likelihood*prior
        post_probs.append(post)
    pred = np.argmax(post_probs)
    return pred


# %%
output = predict(X_train, y_train, X_test[1])
print(output)
print(y_test[1])


# %%
def score(X_train, y_train, X_test, y_test):
    pred = []
    for i in range(X_test.shape[0]):
        pred_label = predict(X_train, y_train, X_test[i])
        pred.append(pred_label)
    pred = np.array(pred)
    accuracy = np.sum(pred==y_test)/y_test.shape[0]
    return accuracy


# %%
print(score(X_train, y_train, X_test, y_test))
