import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
from matplotlib import image as mpimg
from scipy import signal

# a) --------------

# directory of image file
img = mpimg.imread("Fruit.jpg")

# Fast Fourier Transform
imgf = sfft.fft2(img)

# image with Fast Fourier Transform shift
imgf = sfft.fftshift(imgf)

# remove low frequencies
imgf1 = np.zeros((360, 360), dtype=complex)
c = 180
r = 90
for m in range(0, 360):
    for n in range(0, 360):
        if np.sqrt(((m - c) ** 2 + (n - c) ** 2)) > r:
            imgf1[m, n] = imgf[m, n]

# plot image edge
img1 = sfft.ifft2(imgf1)
plt.imshow(np.abs(img1))
plt.show()

# b) --------------

# Gaussian Filter
kernel = np.outer(signal.gaussian(360, 5), signal.gaussian(360, 5))

# Freq Domain Kernel
kf = sfft.fft2(sfft.ifftshift(kernel))
imgf = sfft.fft2(img)
img_b = imgf*kf

# plot [Gaussian blur to the original image]
img1 = sfft.ifft2(img_b)
plt.imshow(np.abs(img1))
plt.show()

# c) --------------

# Discrete Cosine Transform
imgc = sfft.dct((sfft.dct(img, norm='ortho')).T, norm='ortho')

# Inverse Discrete Cosine Transform
img1 = sfft.idct((sfft.idct(imgc, norm='ortho')).T, norm='ortho')

# Removing high frequency components
imgc1 = np.zeros((360, 360))
imgc1[:120, :120] = imgc[:120, :120]
img1 = sfft.idct((sfft.idct(imgc1, norm='ortho')).T, norm='ortho')

# Plot the Scaling image
imgc2 = imgc[0:240, 0:240]
img1 = sfft.idct((sfft.idct(imgc2, norm='ortho')).T, norm='ortho')
plt.imshow(img1)
plt.show()

# d) --------------

# Blocking
# Discrete Cosine Transform
imgc = sfft.dct((sfft.dct(img, norm='ortho')).T, norm='ortho')

# Inverse Discrete Cosine Transform
img1 = sfft.idct((sfft.idct(imgc, norm='ortho')).T, norm='ortho')

# Removing high frequency components
imgc1 = np.zeros((360, 360))
imgc1[:120, :120] = imgc[:120, :120]
img1 = sfft.idct((sfft.idct(imgc1, norm='ortho')).T, norm='ortho')

# Scaling
imgc2 = imgc[0:150, 0:120]
img1 = sfft.idct((sfft.idct(imgc2, norm='ortho')).T, norm='ortho')
plt.imshow(img1)
plt.show()

# Ringing
# Fast Fourier Transform
imgf = sfft.fft2(img)

# image with Fast Fourier Transform shift
imgf = sfft.fftshift(imgf)

# remove high frequencies
imgf1 = np.zeros((480, 480), dtype=complex)
c = 180
r = 50
for m in range(0, 480):
    for n in range(0, 480):
        if np.sqrt(((m - c) ** 2 + (n - c) ** 2)) < r:
            imgf1[m, n] = imgf[m, n]

plt.imshow(np.abs(imgf1))
img1 = sfft.ifft2(imgf1)
plt.imshow(np.abs(img1))
plt.show()
