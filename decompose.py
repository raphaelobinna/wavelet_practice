import pywt
import numpy as np
import matplotlib.pyplot as plt

# Create a simple test signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# Single level decomposition
# coeffs = pywt.dwt(signal, 'db4')
# cA, cD = coeffs  # Approximation and Detail coefficients

# # Reconstruction
# reconstructed = pywt.idwt(cA, cD, 'db4')

# t2 = t[:len(cA)]



# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(t, signal)
# plt.title("Original Signal")

# plt.subplot(3, 1, 2)
# plt.plot(t2, cA, color='green')
# plt.title("Approximation (cA)")

# plt.subplot(3, 1, 3)
# plt.plot(t, reconstructed)
# plt.title("Detail (cD)")

# plt.subplot(3, 1, 4)
# plt.plot(t, reconstructed)
# plt.title("Reconstructed Signal")

# plt.tight_layout()
# plt.show()

# multi level
coeffs = pywt.wavedec(signal, 'db4', level=3)

print(len(coeffs))

rerconstructed = pywt.waverec(coeffs=coeffs, wavelet='db4')