import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# a) --------------

# Function domain [-4pi,4pi]
x_range = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
periodic_range = 2 * np.pi + x_range

# periodic function
def f(x):
    if x < 0 and x >= -np.pi:
        return (x ** 2) + 1
    if x >= 0 and x <= np.pi:
        return x * np.exp(-x)
    if x < -np.pi:
        p = x + (2 * np.pi)
        range1 = f(p)
    if x > np.pi:
        p = x - (2 * np.pi)
        range1 = f(p)

    return range1

# plot the periodic function
y_value = [f(x_value) for x_value in x_range]
plt.plot(periodic_range, y_value)
plt.show()


# b) --------------

# Defining the variables
x1 = sym.symbols('x1')
x2 = sym.symbols('x2')
n = sym.symbols('n', integer=True, positive=True)

# Creating an empty array to store the fourier series expansion
ms = np.empty(150, dtype=object)
x1_range = np.linspace(-4 * np.pi, 4 * np.pi, 1000)

y = np.zeros([150, 1000])

# Defining the equations
eq1 = x1 ** 2 + 1
eq2 = x1 * sym.exp(-1 * x1)

# Defining the general formulas of the fourier series co-efficients
a0 = (1 / sym.pi) * (eq1.integrate((x1, -np.pi, 0)) + eq2.integrate((x1,0,np.pi)))
an = (1 / sym.pi) * (sym.integrate((eq1 * sym.cos(n * x1)), (x1,-1 * np.pi, 0)) + sym.integrate((eq2 * sym.cos(n * x1)), (x1,0,np.pi)))
bn = (1 / sym.pi) * (sym.integrate((eq1 * sym.sin(n * x1)), (x1,-1 * np.pi, 0)) + sym.integrate((eq2 * sym.sin(n * x1)), (x1,0,np.pi)))

print("a0 = ", a0)
print("an = ", an)
print("bn = ", bn)

# First value of the fourier series
ms[0] = a0 / 2

f1 = sym.lambdify(x1, ms[0], 'numpy')
y[0, :] = f1(x1_range)

for m in range(1, 150):
    ms[m] = ms[m - 1] + an.subs(n, m) * sym.cos(m * x1) + bn.subs(n, m) * sym.sin(m * x1)
    f1 = sym.lambdify(x1, ms[m], 'numpy')

    y[m, :] = f1(x1_range)

print("\nThe fourier series = ", ms[1])

# c) --------------

# plot 1st harmonic, 5th harmonic, 150th harmonic
plt.plot(x1_range, y[1, :])
plt.plot(x1_range, y[4, :])
plt.plot(x1_range, y[149, :])

plt.legend(["1", "5", "150"])
plt.show()

# d) --------------

predicted = [y_value]

# Root Mean Square Error for 1st harmonic
actual = [y[1, :]]
MSE = np.square(np.subtract(actual, predicted)).mean()
rsme = math.sqrt(MSE)
print("\nRoot Mean Square Error (1st harmonic) = ", rsme)

# Root Mean Square Error for 5th harmonic
actual = [y[4, :]]
MSE = np.square(np.subtract(actual, predicted)).mean()
rsme = math.sqrt(MSE)
print("\nRoot Mean Square Error (5th harmonic) = ", rsme)

# Root Mean Square Error for 150th harmonic
actual = [y[149, :]]
MSE = np.square(np.subtract(actual, predicted)).mean()
rsme = math.sqrt(MSE)
print("\nRoot Mean Square Error (150th harmonic) = ", rsme)
