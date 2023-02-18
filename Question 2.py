import numpy as np
import matplotlib.pyplot as plt

# Defining the variables
eq_x1 = np.arange(-np.pi, np.pi, 0.01)  # 100Hz
eq_x2 = np.arange(-np.pi, np.pi, 0.1)  # 10Hz

eq_y1 = np.cos(eq_x1) + 0.25 * np.cos(70 * eq_x1)
eq_y2 = np.cos(eq_x2) + 0.25 * np.cos(70 * eq_x2)

# Discrete Fourier Transform
eq_y_dft_1 = np.fft.fft(eq_y1)
eq_y_dft_2 = np.fft.fft(eq_y2)

# plot the Discrete Fourier Transform
plt.plot(eq_x1, eq_y_dft_1)
plt.plot(eq_x2, eq_y_dft_2)
plt.legend(["100Hz", "10Hz"])
plt.title('DFT', pad=20)
plt.show()

# plot the function
plt.plot(eq_x1, eq_y1)
plt.plot(eq_x2, eq_y2)
plt.legend(["100Hz", "10Hz"])
plt.title('Function', pad=20)
plt.show()
