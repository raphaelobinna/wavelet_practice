import pywt
import numpy as np
import matplotlib.pyplot as plt

# List all available wavelets
print(pywt.wavelist())

# List wavelets by family
print(pywt.wavelist('db'))    # Daubechies
print(pywt.wavelist('haar'))  # Haar
print(pywt.wavelist('coif'))  # Coiflets

# Get wavelet properties
wavelet = pywt.Wavelet('db4')
print(f"Family: {wavelet.family_name}")
# print(f"Short name: {wavelet.short_name}")
print(f"Filters length: {wavelet.dec_len}")

# Create noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = clean_signal + noise