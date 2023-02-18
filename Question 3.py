import math
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# a) --------------

# Defining the variables
x = sym.symbols('x')
eq = x * sym.cos(0.5 * x)
x1 = np.arange((-5 * math.pi), (7 * math.pi), 0.01)
f1 = sym.lambdify(x, eq, 'numpy')
y1 = f1(x1)

#  Plot the function
plt.plot(x1, y1)
plt.show()

# b) --------------

# series
def cosine_function(x, n):
    cos_approximate = 0
    for i in range(n):
        coefficient = (-1) ** i
        num = x ** (2 * i)
        denom = math.factorial(2 * i)
        cos_approximate += coefficient * (num / denom)
    return cos_approximate

# x = pi / 2
angle_radian = math.radians(90)
print("\nCos(pi/2)[approximate] = ", cosine_function(angle_radian, 5))


# c) --------------
angles = np.arange(-5 * np.pi, 5 * np.pi, 0.1)
cos = [cosine_function(angle, 60) for angle in angles]  # 60 terms
fig, ax = plt.subplots()
ax.plot(angles, cos)
ax.set_ylim([-3, 3])  # y limitation
plt.show()

# d) --------------
angle_radian = (math.radians(30))
coefficient_term = math.radians(60)
print("\n(pi/3)Cos(pi/6) = ", coefficient_term * cosine_function(angle_radian, 5))


# Discussion
angles_discussion = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
cos_discussion = np.cos(angles_discussion)
cos_discussion_terms = [coefficient_term * cosine_function(angle, 3) for angle in angles_discussion]

fig, ax = plt.subplots()
ax.plot(angles_discussion, cos_discussion)
ax.plot(angles_discussion, cos_discussion_terms)
ax.set_ylim([-5, 5])
ax.legend(['cos Function', 'Taylor Series (3 terms)'])
ax.set_title("Discussion")

plt.show()

print("\nActual Value = ", coefficient_term * math.cos(angle_radian))
