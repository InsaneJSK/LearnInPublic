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
# # This uses the movielens 100k dataset

# %%
import numpy as np
import pandas as pd
import warnings

# %%
warnings.filterwarnings("ignore")

# %%
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep="\t", names=column_names)

# %%
df.head()

# %%
df.shape

# %%
df["user_id"].nunique()

# %%
df["item_id"].nunique()

# %%
movies = pd.read_csv("ml-100k/u.item", sep="|", header=None, encoding="ISO-8859-1")

# %%
movies.head()
movies = movies[[0, 1]]
movies.columns = ["item_id", "title"]

# %%
movies

# %%
df = pd.merge(df, movies, on="item_id")

# %%
df.tail()

# %% [markdown]
# ## Exploratory Data Analysis

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.pairplot(df, kind="hist")

# %%
sns.pairplot(df, kind="scatter")

# %%
sns.pairplot(df, kind="reg")

# %%
ratings = pd.DataFrame(df.groupby('title')["rating"].mean())

# %%
count = pd.DataFrame(df.groupby('title')["rating"].count())

# %%
ratings["num of ratings"] = pd.DataFrame(df.groupby('title').count()['rating'])

# %%
sns.pairplot(movies)

# %%
sns.pairplot(movies, kind="reg")

# %%
sns.pairplot(movies, kind="kde")

# %%
sns.pairplot(movies, kind="hist")

# %%
sns.jointplot(x = "rating", y="num of ratings", data=ratings)

# %%
ratings.sort_values(by="rating", ascending=False)

# %%
sns.jointplot(x = "rating", y="num of ratings", data=ratings, kind="reg")

# %%
sns.jointplot(x = "rating", y="num of ratings", data=ratings, kind="kde")

# %%
moviemat = df.pivot_table(index="user_id", columns="title", values="rating")

# %%
moviemat

# %%
starwars_user_ratings = moviemat["Star Wars (1977)"]
starwars_user_ratings.head()

# %%
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

# %%
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])

# %%
corr_starwars.dropna(inplace=True)

# %%
corr_starwars.head()

# %%
corr_starwars.sort_values(by="Correlation", ascending=False)

# %%
ratings

# %%
corr_starwars = corr_starwars.join(ratings["num of ratings"])

# %%
corr_starwars[corr_starwars['num of ratings']>100].sort_values(by="Correlation", ascending=False)[1:]


# %%
def predict_movie(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings']>100].sort_values(by="Correlation", ascending=False)[1:]
    return predictions


# %%
predict_movie("Titanic (1997)")

# %%
