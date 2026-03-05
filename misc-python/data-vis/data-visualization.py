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
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Visualization

# %% [markdown]
# ## Data Science pipeline
# -------
# - problem statement
# - acquire data
# - exploratory analysis
# - ml modelling
# - visualization to share results

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
#themes
print(plt.style.available)

# %%
plt.style.use('ggplot')

# %%
#Line plot
x = np.arange(10)
y1 = x**2
y2 = 2*x + 3
print(x, y1, y2, sep = "\n")

# %%
plt.plot(x, y1, color = 'red', label="Squares", linestyle="dashed", marker="*")
plt.plot(x, y2, color = 'green', label = "Linear expression", marker = "o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Line Plot")
plt.legend()
plt.show()

# %%
#Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y1, color = 'red', label="Squares", linestyle="dashed", marker="*")
plt.scatter(x, y2, color = 'green', label = "Linear expression", marker = "^")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Line Plot")
plt.legend()
plt.show()

# %%
#Bar Graph
plt.style.use("dark_background")
plt.bar([0,2,4], [10, 20, 15], width=0.5, label="something1", tick_label=[1,2,3])   
plt.bar([0.5,2.5,4.5], [20, 10, 12], width=0.5, label="something2")
plt.title("Something")
plt.legend()
plt.show()

# %%
cats = (1, 2, 3)
weights = [3, 4, 5]
plt.pie(weights, labels = cats, explode=(0.3, 0.2, 0.1), autopct='%1.1f%%', shadow=True)
plt.show()

# %% [markdown]
# ## Histogram

# %%
Xsn = np.random.randn(100)
sigma = 8
mu = 70
X = np.round(Xsn*sigma + mu)
X2 = np.round(Xsn*15 + 40)
print(X, X2)

# %%
plt.style.use('Solarize_Light2')
plt.hist(X, alpha = 0.8, label = "something2")
plt.hist(X2, alpha = 0.8, label = "something")
plt.xlabel("Range")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %% [markdown]
# # Seaborn

# %%
import seaborn as sns
import numpy as np

# %% [markdown]
# ## Barplot

# %%
sns.barplot

# %%
tips_data = sns.load_dataset('tips')

# %%
tips_data

# %%
sns.barplot(x='sex', y='tip', data=tips_data)

# %%
tips_data[tips_data["sex"] == "Female"]["tip"].mean() #height of female

# %%
tips_data[tips_data["sex"] == "Male"]["tip"].mean() #height of male

# %%
sns.barplot(x='sex', y='tip', data=tips_data, estimator=np.std)

# %% [markdown]
# ## Countplot

# %%
sns.countplot(x="sex", data=tips_data)

# %%
sns.countplot(y="sex", data=tips_data)  #You can only either pass x or y

# %%
sns.countplot(x="day", hue='sex', data=tips_data)

# %%
sns.countplot(y="day", hue='sex', data=tips_data)

# %%
sns.countplot(x="sex", hue='day', data=tips_data)

# %% [markdown]
# ## Boxplot

# %%
sns.boxplot(x="sex", y="tip", data=tips_data)

# %%
sns.boxplot(x="day", y="tip", hue="sex", data=tips_data)

# %%
sns.violinplot(x="sex", y="tip", data=tips_data)

# %%
sns.violinplot(x="day", y="tip", hue="sex", data=tips_data)

# %% [markdown]
# ## Distribution Plot

# %%
sns.distplot(tips_data["tip"], kde= True)

# %%
sns.displot(tips_data["tip"]) #Kernel Density Estimation is used here

# %% [markdown]
# ## KDE plot

# %%
sns.kdeplot(tips_data["tip"])

# %% [markdown]
# ## Jointplot

# %%
sns.jointplot(x = "total_bill", y="tip", data=tips_data)

# %%
sns.jointplot(x = "total_bill", y="tip", data=tips_data, kind="hex")

# %%
sns.jointplot(x = "total_bill", y="tip", data=tips_data, kind="kde")

# %%
sns.jointplot(x = "total_bill", y="tip", data=tips_data, kind="reg")

# %% [markdown]
# ## Pairplot

# %%
sns.pairplot(tips_data)

# %%
sns.pairplot(tips_data, hue="sex")

# %% [markdown]
# ## Heatmap

# %%
flights = sns.load_dataset('flights')

# %%
flights

# %%
flights_pivot = flights.pivot_table(index="month", columns="year", values="passengers")

# %%
sns.heatmap(flights_pivot, cmap="coolwarm")

# %%
tips_data.corr(numeric_only=True)

# %%
sns.heatmap(tips_data.corr(numeric_only=True))
