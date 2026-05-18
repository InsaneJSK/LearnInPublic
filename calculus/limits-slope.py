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
# ## Limits
#

# %%
import sympy as sp
x = sp.symbols('x')
f = 3*x*(x-2)/(x-2)
f

# %%
x = 2
h = 0.00001

y_right = (3*((x+h)-2))/((x+h)-2)
y_left = (3*((x-h)-2))/((x-h)-2)

print(f"Right-hand limit: {y_right}")
print(f"Left-hand limit: {y_left}")

if round(y_right, 5) == round(y_left, 5):
    print("The limit exists and is:", round(y_right, 5))
else:
    print("The limit does not exist.")

# %%
x = sp.symbols('x')
eq = x**2 - 4
print(eq)
print(f"X = {sp.solve(eq, x)}")

# %%
y = 3*(x**2)/(x**2-4)
y

# %%
x = 2
h = 0.00001
y_right = 3*(((x+h)**2)/((x+h)**2-4))
y_left = 3*(((x-h)**2)/((x-h)**2-4))
print(f"y_right: {y_right}")
print(f"y_left: {y_left}")
if round(y_right, 5) != round(y_left, 5):
    print("The limit doesn't exist at x =", x)
else:
    print("The limit exists at x =", x, "and is approximately:", round(y_right, 5))

# %%
y

# %%
for x_value in [10e0, 10e1, 10e2, 10e3, 10e4]:
    limit_value = 3*(x_value**2)/(x_value**2-4)
    print(f"Limit as x approaches {x_value}: {limit_value}")

# %%
import matplotlib.pyplot as plt
import numpy as np
x_values = np.linspace(-10, 10, 400)
y_values = 3*(x_values**2)/(x_values**2-4)
plt.axis([-10, 10, -10, 10])
plt.plot([0], [0], 'ro')  # Mark the point (0, 0) with a red dot
plt.title("Graph of y = 3*(x^2)/(x^2-4)")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black', lw=0.5, ls='--')  # Add x-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Add y-axis
plt.plot(x_values, y_values)
plt.show()

# %%
# Graph using sympy
x = sp.symbols('x')
y = 3*(x**2)/(x**2-4)
# Critical values for y
x1 = 2
x2 = -2
#x1
right = sp.limit(y, x, x1, dir='+')
left = sp.limit(y, x, x1, dir='-')
print(f"Right-hand limit at x = {x1}: {right}")
print(f"Left-hand limit at x = {x1}: {left}")
#x2
right = sp.limit(y, x, x2, dir='+')
left = sp.limit(y, x, x2, dir='-')
print(f"Right-hand limit at x = {x2}: {right}")
print(f"Left-hand limit at x = {x2}: {left}")
#infinite
right = sp.limit(y, x, sp.oo)
left = sp.limit(y, x, -sp.oo)
print(f"Limit as x approaches infinity: {right}")
print(f"Limit as x approaches negative infinity: {left}")

# %% [markdown]
# ## Slope

# %%
x_value = 1
y_value = x_value**2

h = 0.3
y_value_2 = (x_value + h)**2

zoom = 10
xmin = x_value - zoom
xmax = x_value + zoom
ymin = y_value - zoom
ymax = y_value + zoom

x = np.linspace(xmin, xmax, 400)
y = x**2

slope = (y_value_2 - y_value) / h
print(f"Slope of the secant line: {slope}")

plt.axis([xmin, xmax, ymin, ymax])
plt.plot(x, y, label='y = x^2')
plt.plot([x_value, x_value + h], [y_value, y_value_2], 'ro-', label='Secant Line')
plt.axhline(0, color='black', lw=0.5, ls='--')  # Add x-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Add y-axis
plt.title("Graph of y = x^2 with Secant Line")
plt.show()

# %%
x_value = 1
y_value = x_value**2

h = 0.3
y_value_2 = (x_value + h)**2

for a in range(1, 11):
    h = 10**(-a)
    y_value_2 = (x_value + h)**2
    slope = (y_value_2 - y_value) / h
    print(f"h: {h}, Slope of the secant line: {slope}")

# %%
x_value = 1
y_value = x_value**3

h = 0.3
y_value_2 = (x_value + h)**3

zoom = 10
xmin = x_value - zoom
xmax = x_value + zoom
ymin = y_value - zoom
ymax = y_value + zoom

x = np.linspace(xmin, xmax, 400)
y = x**3

slope = (y_value_2 - y_value) / h
print(f"Slope of the secant line: {slope}")

plt.axis([xmin, xmax, ymin, ymax])
plt.plot(x, y, label='y = x^3')
plt.plot([x_value, x_value + h], [y_value, y_value_2], 'ro-', label='Secant Line')
plt.axhline(0, color='black', lw=0.5, ls='--')  # Add x-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Add y-axis
plt.title("Graph of y = x^3 with Secant Line")
plt.show()

# %%
x_value = 1
y_value = x_value**3

h = 0.3
y_value_2 = (x_value + h)**3

for a in range(1, 11):
    h = 10**(-a)
    y_value_2 = (x_value + h)**3
    slope = (y_value_2 - y_value) / h
    print(f"h: {h}, Slope of the secant line: {slope}")

# %%
