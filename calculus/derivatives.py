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

# %% [markdown]
# ### Implicit differentiation

# %%
import sympy as sp
x, y = sp.symbols('x y')
eq = x**2 + y**2 - 25
derivative = sp.idiff(eq, y, x)
eq

# %%
derivative

# %%
sp.plot_implicit(eq)

# %%
x_val, y_val = 3, 4
ans_x = derivative.subs({x: x_val, y: y_val})
ans_y = ans_x.subs(y, y_val)
print(ans_y)

# %%
x_val = 3
x_part = eq.subs(x, x_val)
y_part = sp.Eq(x_part, 0)
y_val = sp.solve(y_part, y)
print(f"At x = {x_val}, y can be: {y_val}")

# %%
for y_loop in y_val:
    ans_x = derivative.subs({x: x_val})
    ans_y = ans_x.subs(y, y_loop)
    print(f"At x = {x_val} and y = {y_loop}, the slope of the tangent line is: {ans_y}")

# %%
