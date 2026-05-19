# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Derivatives

# %%
import sympy as sp
x, y = sp.symbols('x y')
eq = x**3
derivative = sp.diff(eq, x)
print(f"Derivative of {eq} with respect to x is: {derivative}")

# %%
answer = derivative.subs(x, 2)
print(f"The slope of the tangent line at x = 2 is: {answer}")

# %% [markdown]
# ### Power Rule

# %%
x, y = sp.symbols("x y")
eq = x**3
eq

# %%
eq.diff(x)

# %% [markdown]
# ### Chain Rule

# %%
eq = (x**2 + 8)**4
eq

# %%
eq.diff(x)

# %%
eq = sp.sqrt(x**2 + 8)
eq

# %%
eq.diff(x)

# %%
eq = sp.sin(x**3)
eq

# %%
eq.diff(x)

# %% [markdown]
# ### Product Rule

# %%
eq = (x**2 + 8)*(x**3 + 5)
eq

# %%
eq.diff(x)

# %% [markdown]
# ### Quotient Rule

# %%
eq = (x**3 - 1)/(x**2 - 1)
eq

# %%
eq.diff(x)

# %%
eq.diff(x).subs(x, 2)

# %%
