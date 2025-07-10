import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
import pandas as pd

print(pywt.wavelist('db')) 
wavelet = pywt.Wavelet('db8')
print(f"Filters length: {wavelet.dec_len}")
print(pywt.wavelist('coif'))

wavelet = pywt.Wavelet('bior2.2')
print(f"Family name {wavelet.family_name}")

t = np.linspace(0,2,1000)
signal_5Hz = np.sin(2*np.pi*5*t)
signal_20Hz = np.sin(2*np.pi*20*t)
combined_signal = signal_5Hz + signal_20Hz

# plt.figure(figsize=(10, 4))
# plt.plot(t, combined_signal, label='5Hz + 20Hz')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Signal composed of 5 Hz and 20 Hz Sine Waves')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# single leevel

coeffs = pywt.dwt(combined_signal, 'db4')
print(len(coeffs))
cA, cD = coeffs

print("Length of original signal:", len(combined_signal))
print("Length of approximation coefficients (cA):", len(cA))
print("Length of detail coefficients (cD):", len(cD))

reconstructed = pywt.idwt(cA,cD,'db4')

min_len = np.min([len(combined_signal), len(reconstructed)])

reconstruction_error = np.max(np.abs(combined_signal[:min_len] -reconstructed[:min_len]))
print("Max reconstruction error:", reconstruction_error)


t_full = t
t_half = np.linspace(0, 2, len(cA)) 

# # Create 2x2 subplot figure
# plt.figure(figsize=(12, 8))

# # Top left: Original signal
# plt.subplot(2, 2, 1)
# plt.plot(t_full, combined_signal, color='black')
# plt.title("Original Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")

# # Top right: Approximation coefficients
# plt.subplot(2, 2, 2)
# plt.plot(t_half, cA, color='green')
# plt.title("Approximation Coefficients (cA)")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")

# # Bottom left: Detail coefficients
# plt.subplot(2, 2, 3)
# plt.plot(t_half, cD, color='red')
# plt.title("Detail Coefficients (cD)")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")

# # Bottom right: Reconstructed signal
# plt.subplot(2, 2, 4)
# plt.plot(t_full[:len(reconstructed)], reconstructed[:len(t_full)], color='blue')
# plt.title("Reconstructed Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()


# Time vector: 0 to 2 seconds, 1000 points
t = np.linspace(0, 2, 1000)

# a) Signal A: Pure sine wave at 10 Hz
signal_A = np.sin(2 * np.pi * 10 * t)

# b) Signal B: Step function (0 until 1 sec, then 1)
signal_B = np.heaviside(t - 1, 1)  # 0 for t<1, 1 for t>=1

# c) Signal C: Chirp signal, frequency from 1 to 50 Hz over 2 seconds
signal_C = chirp(t, f0=1, f1=50, t1=2, method='linear')

# Plot all three signals
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(t, signal_A)
# plt.title("Signal A: 10 Hz Pure Sine Wave")
# plt.ylabel("Amplitude")
# plt.grid(True)

# plt.subplot(3, 1, 2)
# plt.plot(t, signal_B)
# plt.title("Signal B: Step Function (Heaviside)")
# plt.ylabel("Amplitude")
# plt.grid(True)

# plt.subplot(3, 1, 3)
# plt.plot(t, signal_C)
# plt.title("Signal C: Chirp Signal (1 to 50 Hz)")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.grid(True)

# plt.tight_layout()
# plt.show()


cA, cD = pywt.dwt(signal_A, 'haar')

max_abs_a = np.max(np.abs(cA))
max_abs_d = np.max(np.abs(cD))



# a) Create signal with 1024 points
t = np.linspace(0, 2, 1024)
signal_many = np.sin(2 * np.pi * 10 * t)

# b) 4-level decomposition using 'db6'
coeffs_many = pywt.wavedec(signal_many, 'db6', level=4)

# b) Print structure: length of each coefficient array
print("Wavelet Coefficients Structure (level=4):")
for i, coeff in enumerate(coeffs_many):
    label = "cA4" if i == 0 else f"cD{4 - i + 1}"
    print(f"{label}: length = {len(coeff)}")

# c) Total number of coefficients
total_coeffs = sum(len(c) for c in coeffs_many)
print(f"\nTotal number of coefficients: {total_coeffs}")

# d) Reconstruct and verify error
reconstructed = pywt.waverec(coeffs_many, 'db6')
min_len = min(len(signal_many), len(reconstructed))
reconstruction_error = np.max(np.abs(signal_many[:min_len] - reconstructed[:min_len]))
print(f"\nMax reconstruction error: {reconstruction_error:.6e}")


# Generate noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = clean_signal + noise

# Define wavelets
wavelets = ['haar', 'db4', 'db8', 'coif2']
energy_data = {}

# Compute energy at each detail level for each wavelet
for w in wavelets:
    coeffs = pywt.wavedec(noisy_signal, wavelet=w, level=3)
    energy_data[w] = {
        f"D{i}": np.sum(coeffs[-i] ** 2) for i in range(1, 4)
    }

# # Convert to DataFrame
# df = pd.DataFrame(energy_data).T
# df.index.name = 'Wavelet'
# df = df[['D1', 'D2', 'D3']]  # Ensure consistent column order

# # Plot grouped bar chart
# df.plot(kind='bar', figsize=(10, 6), colormap='viridis')

# plt.title("Energy Distribution Across Detail Levels (D1, D2, D3)")
# plt.xlabel("Wavelet")
# plt.ylabel("Energy")
# plt.xticks(rotation=0)
# plt.grid(axis='y')
# plt.legend(title="Detail Level")
# plt.tight_layout()
# plt.show()


signal = np.zeros(512)
signal[0:256] = 1.0 

all_modes = [
'symmetric',
'periodization',
'zero']

result = {}

for m in all_modes:
    cA2, cD2, cD1 = pywt.wavedec(signal, 'db4', mode=m, level=2)

    lowest_ten = cD1[len(cD1)-10:len(cD1)]
    highest_ten = cD1[:10]

    result[m] = {
        "first_10": cD1[:10],
        "last_10": cD1[-10:]
    }

# c) Print and compare
for mode in result:
    print(f"\nMode: {mode}")
    print("First 10 cD1 coefficients:", np.round(result[mode]['first_10'], 4))
    print("Last 10 cD1 coefficients: ", np.round(result[mode]['last_10'], 4))


t = np.linspace(0, 1, 1000)
high_signal = np.sin(2 * np.pi * 50 * t)
low_signal = np.sin(2 * np.pi * 5 * t)

full_signal = high_signal+low_signal

coeffs = pywt.wavedec(full_signal, 'db4', level=4)

# first version
first_version = coeffs.copy()

for i in range(1, len(first_version)):
    first_version[i] = np.zeros_like(first_version[i])

first_version_reconstruct = pywt.waverec(first_version, wavelet='db4')

# second version
second_version = coeffs.copy()

second_version[0] = np.zeros_like(second_version[0])

second_version_reconstruct = pywt.waverec(second_version, wavelet='db4')

# third version
third_version = coeffs.copy()

third_version[-1]  = np.zeros_like(third_version[-1])

third_version_reconstruct = pywt.waverec(third_version, wavelet='db4')

min_len = min(len(full_signal), len(first_version_reconstruct), len(second_version_reconstruct), len(third_version_reconstruct))
full_signal = full_signal[:min_len]
v1_recon = first_version_reconstruct[:min_len]
v2_recon = second_version_reconstruct[:min_len]
v3_recon = third_version_reconstruct[:min_len]

# d) Plot all 4 signals
# plt.figure(figsize=(12, 10))

# plt.subplot(4, 1, 1)
# plt.plot(t[:min_len], full_signal)
# plt.title("Original Signal (Low + High Frequency)")

# plt.subplot(4, 1, 2)
# plt.plot(t[:min_len], v1_recon)
# plt.title("Version 1: Only Approximation (Low Frequency)")

# plt.subplot(4, 1, 3)
# plt.plot(t[:min_len], v2_recon)
# plt.title("Version 2: Only Details (High + Mid Frequency)")

# plt.subplot(4, 1, 4)
# plt.plot(t[:min_len], v3_recon)
# plt.title("Version 3: All but Level-1 Detail (High Freq Removed)")

# plt.tight_layout()
# plt.show()

signal_lengths = [64, 256, 1024, 4096]
wavelet = 'db4'


def optimal_decomposition_level(signal_len,wavelet_name):
    
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(signal_len, wavelet.dec_len)

    return max_level



for length in signal_lengths:
    level = optimal_decomposition_level(length, wavelet)
    print(f"Signal length: {length} → Optimal decomposition level: {level}")