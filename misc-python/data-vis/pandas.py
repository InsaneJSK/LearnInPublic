# ---
# jupyter:
#   jupytext:
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
user_data = {
    "MarksA" : np.random.randint(1, 100, 5),
    "MarksB" : np.random.randint(1, 100, 5),
    "MarksC" : np.random.randint(1, 100, 5),
}

# %%
user_data

# %%
df = pd.DataFrame(user_data, dtype="float")
print(df)

# %%
df.head(3)

# %%
df.to_csv("marks.csv")

# %%
mydf = pd.read_csv("marks.csv")

# %%
mydf.head()

# %%
mydf.describe()

# %%
mydf.tail(3)

# %%
mydf.iloc[3, 1] #same as df.iloc[3][1]
#You can use either of indexing or slicing in this exact way

# %%
mydf.sort_values(by=["MarksA", "MarksC"], ascending=True)

# %%
mydf.shape

# %%
mydfar = mydf.values
mydfar = mydfar[:, 1:] #literally just dropping a row could be done using pd too

# %%
nmydf = pd.DataFrame(mydfar, dtype='int32', columns=["P", "C", "M"])

# %%
nmydf

# %% [markdown]
# ## MNIST Dataset

# %%
df = pd.read_csv("mnist_train.csv")

# %%
df.head()

# %%
data = df.values

# %%
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# %%
print(y)


# %%
def imageiter(end, start = 0):
    while start < end:
        yield start
        start += 1


# %%
imagegen = imageiter((X.shape[0]))

# %%

import matplotlib.pyplot as plt


# %%
i = next(imagegen)
plt.imshow(X[i, :].reshape(28, 28), cmap = "grey")
plt.title(f"Label {y[i]}")
plt.show()


# %%
plt.figure(figsize=(10, 10))
for k in range(1, 626):
    plt.subplot(25, 25, k)
    i = next(imagegen)
    plt.imshow(X[i, :].reshape(28, 28), cmap = "grey")
    plt.axis("off")
plt.show()

# %%
## Movie Dataset

# %%
df = pd.read_csv("movie_metadata.csv")

# %%
list(df.columns)

# %%
df.describe()

# %%
titles = list(df.get("movie_title"))

# %%
dicti = {}
for t in titles:
    if len(t) in dicti.keys():
        dicti[len(t)] += 1
    else:
        dicti[len(t)] = 1


# %%
print(dicti)

# %%
plt.scatter(np.array(list(dicti.keys())), np.array(list(dicti.values())))
plt.title("Movie Name length freq")
plt.xlabel("Number of characters")
plt.ylabel("Number of movies")
plt.show()

# %%
