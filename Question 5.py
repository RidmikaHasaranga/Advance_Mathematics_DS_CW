import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import logistic

# Defining the variables
x = sym.symbols('x')
f = 1/(1 + sym.exp(-x))

# a) --------------
f1 = sym.lambdify(x, f, 'numpy')
x1 = np.arange(-40, 40, 0.01)
y1 = f1(x1)
plt.plot(x1, y1)
plt.show()

# b) --------------
eq_diff = sym.diff(f)
f2 = sym.lambdify(x, eq_diff, 'numpy')
x2 = np.arange(-40, 40, 0.1)
y2 = f2(x2)
plt.plot(x2, y2)
plt.show()

# c) a. --------------
eq3 = sym.sin(sym.sin(2 * x))
f3 = sym.lambdify(x, eq3, 'numpy')
x3 = np.arange(-40, 40, 0.1)
y3 = f3(x3)
plt.plot(x3, y3)
plt.show()

# c) b. --------------
eq4 = (-x ** 3) - 2 * (x ** 2) + 3 * x + 10
f4 = sym.lambdify(x, eq4, 'numpy')
x4 = np.arange(-40, 40, 0.1)
y4 = f4(x4)
plt.plot(x4, y4)
plt.show()

# c) c. --------------
eq5 = sym.exp(-0.8 * x)
f5 = sym.lambdify(x, eq5, 'numpy')
x5 = np.arange(-40, 40, 0.1)
y5 = f5(x5)
plt.plot(x5, y5)
plt.show()

# c) d. --------------
eq6 = (x ** 2 * sym.cos(sym.cos(2 * x))) - 2 * sym.sin(sym.sin(x - math.pi / 3))
f6 = sym.lambdify(x, eq6, 'numpy')
x6 = np.arange(-40, 40, 0.1)
y6 = f6(x6)
plt.plot(x6, y6)
plt.show()

# c) e. --------------

# Plot the graph
y7 = sym.Piecewise((2 * sym.cos(x + np.pi / 6), x < 0), (x * sym.exp(-0.4 * x ** 2), x < np.pi), (0, True))
sym.plot(y7, (x, -40, 40))

# d) --------------

# Defining the range of x axis
x = np.arange(-40, 40, 0.1)

# logistic function
def logistic_1(x):
    return 1 / (1 + np.exp(-x))

#  Apply the logistic function for c) a. --------------
y = logistic_1(np.sin(np.sin(2 * x)))
plt.plot(x, y)
plt.show()

# Apply the logistic function for c) b. --------------
y = logistic_1((-x ** 3) - 2 * (x ** 2) + 3 * x + 10)
plt.plot(x, y)
plt.show()

#  Apply the logistic function for c) c. --------------
y = logistic_1(np.exp(-0.8 * x))
plt.plot(x, y)
plt.show()

#  Apply the logistic function for c) d. --------------
y = logistic_1((x ** 2 * np.cos(np.cos(2 * x))) - 2 * np.sin(np.sin(x - math.pi / 3)))
plt.plot(x, y)
plt.show()

# Apply the logistic function for c) e. --------------

# Define new x (because error occur)
x2 = sym.symbols('x2')

# logistic function using sympy
def logistic_2(x2):
    return 1 / (1 + sym.exp(-1 * x2))

function1 = sym.Piecewise((2 * sym.cos(x2 + np.pi / 6), x2 < 0), (x2 * sym.exp(-0.4 * x2 ** 2), x2 < np.pi), (0, True))

y = logistic_2(function1)
sym.plot(y, (x2, -40, 40))
