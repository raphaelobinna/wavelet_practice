import pywt
import numpy as np
import matplotlib.pyplot as plt

# Create noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = clean_signal + noise

coeffs = pywt.wavedec(noisy_signal,'db4', level=4)


threshold = 10

coeff_threshold = coeffs.copy()

for i in range(1, len(coeffs)):
    coeff_threshold[i] = pywt.threshold(coeff_threshold[i], threshold, mode='soft')


reconstructed = pywt.waverec(coeff_threshold, 'db4')


# Plot in separate subplots
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, clean_signal, color='green')
plt.title('Clean Signal')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, color='gray')
plt.title('Noisy Signal')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed[:len(t)], color='blue')
plt.title('Denoised Signal (Wavelet)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()