import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, stft
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
from skimage.draw import disk, rectangle
from skimage.util import random_noise

# a) Time vector
t = np.linspace(0, 2, 1000)

# Clean signal components
signal_clean = (
    np.sin(2 * np.pi * t) +
    0.5 * np.sin(8 * np.pi * t) +
    0.25 * np.sin(16 * np.pi * t)
)

# Add Gaussian noise with σ = 0.3
noise = np.random.normal(0, 0.3, size=len(t))
signal_noisy = signal_clean + noise

# # Plot both signals
# plt.figure(figsize=(10, 5))
# plt.plot(t, signal_clean, label="Clean Signal", linewidth=2)
# plt.plot(t, signal_noisy, label="Noisy Signal", alpha=0.7)
# plt.title("Clean vs. Noisy Signal")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# b) Perform DWT
wavelet = 'db4'
level = 5
coeffs = pywt.wavedec(signal_noisy, wavelet=wavelet, level=level)

threshold = 0.4

def threshold_coeffs(coeffs, threshold, mode='soft', adaptive=False):
    new_coeffs = [coeffs[0]]  # keep approximation unchanged
    for i, cD in enumerate(coeffs[1:], start=1):
        if adaptive:
            thr = threshold / (i ** 0.5)  # adaptive per level
        else:
            thr = threshold
        new_coeffs.append(pywt.threshold(cD, thr, mode=mode))
    return new_coeffs

# Apply 3 methods
hard_coeffs = threshold_coeffs(coeffs, threshold, mode='hard')
soft_coeffs = threshold_coeffs(coeffs, threshold, mode='soft')
adapt_coeffs = threshold_coeffs(coeffs, threshold, mode='soft', adaptive=True)

# Reconstruct
denoised_hard = pywt.waverec(hard_coeffs, wavelet)
denoised_soft = pywt.waverec(soft_coeffs, wavelet)
denoised_adaptive = pywt.waverec(adapt_coeffs, wavelet)

# Clip to match length , padding may occur
min_len = min(len(t), len(denoised_hard))
signal_clean = signal_clean[:min_len]
signal_noisy = signal_noisy[:min_len]
denoised_hard = denoised_hard[:min_len]
denoised_soft = denoised_soft[:min_len]
denoised_adaptive = denoised_adaptive[:min_len]

# c) Evaluate using SNR
def compute_snr(clean, denoised):
    noise = clean - denoised
    return 20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(noise))

snr_hard = compute_snr(signal_clean, denoised_hard)
snr_soft = compute_snr(signal_clean, denoised_soft)
snr_adaptive = compute_snr(signal_clean, denoised_adaptive)

print(f"SNR (Hard):     {snr_hard:.2f} dB")
print(f"SNR (Soft):     {snr_soft:.2f} dB")
print(f"SNR (Adaptive): {snr_adaptive:.2f} dB")

# d) Plot all signals
# plt.figure(figsize=(14, 10))

# plt.subplot(5, 1, 1)
# plt.plot(t[:min_len], signal_clean, label="Clean")
# plt.title("Original Clean Signal")
# plt.grid()

# plt.subplot(5, 1, 2)
# plt.plot(t[:min_len], signal_noisy, label="Noisy")
# plt.title("Noisy Signal (σ = 0.3)")
# plt.grid()

# plt.subplot(5, 1, 3)
# plt.plot(t[:min_len], denoised_hard, label=f"Hard Denoised (SNR={snr_hard:.2f} dB)")
# plt.title("Denoised (Hard Thresholding)")
# plt.grid()

# plt.subplot(5, 1, 4)
# plt.plot(t[:min_len], denoised_soft, label=f"Soft Denoised (SNR={snr_soft:.2f} dB)")
# plt.title("Denoised (Soft Thresholding)")
# plt.grid()

# plt.subplot(5, 1, 5)
# plt.plot(t[:min_len], denoised_adaptive, label=f"Adaptive Denoised (SNR={snr_adaptive:.2f} dB)")
# plt.title("Denoised (Adaptive Thresholding)")
# plt.grid()

# plt.tight_layout()
# plt.show()

# Time vector: 0 to 2 seconds, 1000 points
t = np.linspace(0, 2, 1000)
f2=1
f1=10

# c) Signal C: Chirp signal, frequency from 1 to 10 Hz over 2 seconds
signal_A = chirp(t, f0=1, f1=10, t1=2, method='linear')

# b) Impulse train (5 impulses at random locations)
np.random.seed(0)
impulse_indices = np.random.choice(len(t), 5, replace=False)
signal_B = np.zeros_like(t)
signal_B[impulse_indices] = 1.0

# a) Signal A: Pure sine wave at 10 Hz
signal_C = np.sin(2 * np.pi * f1 * t* (1+ 0.5* np.sin(2 * np.pi * f2 *t)))





def extract_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    features = {
        "energy": {},
        "mean": {},
        "variance": {},
        "skewness": {},
        "kurtosis": {},
        "entropy": {}
    }

    # Only detail coefficients (cD1, cD2, ...)
    for i, cD in enumerate(coeffs[1:], start=1):
        level_key = f"cD{i}"
        features["energy"][level_key] = np.sum(cD**2)
        features["mean"][level_key] = np.mean(cD)
        features["variance"][level_key] = np.var(cD)
        features["skewness"][level_key] = skew(cD)
        features["kurtosis"][level_key] = kurtosis(cD)
        
        # Normalized for entropy
        prob = np.abs(cD) / np.sum(np.abs(cD))
        features["entropy"][level_key] = entropy(prob, base=2)

    return features

# ========== RUN AND DISPLAY FEATURES ========== #

features_A = extract_features(signal_A)
features_B = extract_features(signal_B)
features_C = extract_features(signal_C)

def display_features(name, features):
    print(f"\n📊 Features for {name}:")
    for category, stats in features.items():
        print(f"  {category.capitalize()}:")
        for level, value in stats.items():
            print(f"    {level}: {value:.4f}")

display_features("Signal A (Chirp)", features_A)
display_features("Signal B (Impulse Train)", features_B)
display_features("Signal C (AM Signal)", features_C)

def flatten_features(feature_dict):
    flat = {}
    for category, values in feature_dict.items():
        for level, val in values.items():
            flat[f"{category}_{level}"] = val
    return flat

# Flatten each signal's features
flat_A = flatten_features(features_A)
flat_B = flatten_features(features_B)
flat_C = flatten_features(features_C)

# === Step 2: Create Feature Matrix === #
df = pd.DataFrame([flat_A, flat_B, flat_C], index=["Chirp", "Impulse", "AM"])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

print("\n📊 Feature Matrix (Standardized):")
print(df_scaled.round(2))

# === Step 3: Identify Most Discriminative Features === #
# Using variance across signals
variances = df_scaled.var(axis=0).sort_values(ascending=False)

# Top 5 most variable features
top_features = variances.head(5).index.tolist()

print("\n📈 Top Discriminative Features:")
for feat in top_features:
    print(f"  {feat} (Variance = {variances[feat]:.2f})")

# # === Step 4: Plot These Features === #
# plt.figure(figsize=(10, 6))
# for feat in top_features:
#     sns.lineplot(x=df_scaled.index, y=df_scaled[feat], marker='o', label=feat)

# plt.title("Most Discriminative Features Across Signal Types")
# plt.ylabel("Standardized Feature Value")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


t = np.linspace(0, 2, 2048)

signal_clean = (
    np.sin(2 * np.pi * t) +
    0.5 * np.sin(8 * np.pi * t) +
    0.25 * np.sin(16 * np.pi * t)
)

def compress_signal(signal, wavelet='db4', level=5, keep_percent=10):
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Flatten all coefficients to sort globally
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    total_coeffs = coeff_array.size
    
    # Calculate number of coefficients to keep
    keep_n = int((keep_percent / 100) * total_coeffs)

    # Find threshold
    threshold = np.partition(np.abs(coeff_array.flatten()), -keep_n)[-keep_n]
    
    # Apply threshold
    compressed_array = np.where(np.abs(coeff_array) >= threshold, coeff_array, 0)

    # Reconstruct from thresholded array
    compressed_coeffs = pywt.array_to_coeffs(compressed_array, coeff_slices, output_format='wavedec')
    reconstructed = pywt.waverec(compressed_coeffs, wavelet)

    return reconstructed, compressed_array, total_coeffs, keep_n

# ========== SNR Calculation ========== #
def compute_snr(original, reconstructed):
    noise = original - reconstructed
    return 20 * np.log10(np.linalg.norm(original) / np.linalg.norm(noise))

# ========== Run Experiments for Various Compression Ratios ========== #
ratios = [10, 25, 50, 75, 90]
results = []

for r in ratios:
    reconstructed, compressed_arr, total, kept = compress_signal(signal_clean, keep_percent=r)
    min_len = min(len(signal_clean), len(reconstructed))
    
    mse = mean_squared_error(signal_clean[:min_len], reconstructed[:min_len])
    snr = compute_snr(signal_clean[:min_len], reconstructed[:min_len])
    compression_ratio = (total - kept) / total

    results.append((r, compression_ratio, mse, snr))

# ========== Show Results ========== #
print(f"{'Kept %':<8} {'Compression Ratio':<20} {'MSE':<15} {'SNR (dB)':<10}")
for r, comp_ratio, mse, snr in results:
    print(f"{r:<8} {comp_ratio:<20.4f} {mse:<15.6f} {snr:<10.2f}")

# # ========== Plot Original vs Compressed ========== #
# plt.figure(figsize=(12, 6))
# for i, r in enumerate(ratios[:3]):  # Plot only first 3 for clarity
#     reconstructed, *_ = compress_signal(signal_clean, keep_percent=r)
#     plt.plot(t[:len(reconstructed)], reconstructed, label=f'{r}% Kept', alpha=0.7)

# plt.plot(t, signal_clean, label='Original', linewidth=2, linestyle='--', color='black')
# plt.title("Wavelet Compression (First 3 Levels)")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Sampling setup
fs = 1000  # Sampling rate (Hz)
t = np.linspace(0, 3, 3 * fs, endpoint=False)

# Time segments
t1 = t[:fs]     # 0–1s
t2 = t[fs:2*fs] # 1–2s
t3 = t[2*fs:]   # 2–3s

# Individual components
sig1 = np.sin(2 * np.pi * 5 * t1)
sig2 = np.sin(2 * np.pi * 15 * t2)
sig3 = np.sin(2 * np.pi * 25 * t3)

signal = np.concatenate([sig1, sig2, sig3])


scales = np.arange(1, 128)

# Use Complex Morlet wavelet
wavelet = 'cmor1.5-1.0'  # (bandwidth-frequency product - center freq)

# === STFT ===
f_stft, t_stft, Zxx = stft(signal, fs=fs, window='hann', nperseg=256, noverlap=128)

# === CWT ===
cwt_coeffs, freqs_cwt = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

# === PLOT BOTH ===
# plt.figure(figsize=(14, 6))

# # --- STFT ---
# plt.subplot(1, 2, 1)
# plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud', cmap='jet')
# plt.title("STFT Spectrogram")
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.colorbar(label="Magnitude")
# plt.ylim(0, 50)

# # --- CWT ---
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(cwt_coeffs), extent=[0, 3, freqs_cwt[-1], freqs_cwt[0]],
#            aspect='auto', cmap='jet')
# plt.title("CWT Scalogram (cmor)")
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.colorbar(label="Magnitude")
# plt.ylim(0, 50)

# plt.tight_layout()
# plt.show()


# Step 1: Create blank 128x128 image
image = np.zeros((128, 128), dtype=np.float32)

# Step 2: Add a circle at the center
center = (64, 64)
radius = 30
rr, cc = disk(center, radius)
image[rr, cc] = 1.0  # white circle on black background

# Step 3: Add Gaussian noise (mean=0, var=0.01)
noisy_image = random_noise(image, mode='gaussian', var=0.01)

wavelet = 'db2'
coeffs2 = pywt.dwt2(noisy_image, wavelet)
LL, (LH, HL, HH) = coeffs2

threshold = 0.05
LH_d = pywt.threshold(LH, threshold, mode='soft')
HL_d = pywt.threshold(HL, threshold, mode='soft')
HH_d = pywt.threshold(HH, threshold, mode='soft')

# Step 5: Reconstruct denoised image
coeffs_denoised = (LL, (LH_d, HL_d, HH_d))
denoised_image = pywt.idwt2(coeffs_denoised, wavelet)

coeffs_multilevel = pywt.wavedec2(noisy_image, wavelet, level=3)

# Visualize LL and first-level details
LL3, details = coeffs_multilevel[0], coeffs_multilevel[1:]

fs = 1000
t = np.linspace(0, 2, 2 * fs)

# Create frequency-modulated signal (chirp: 5Hz → 50Hz)
signal = chirp(t, f0=5, f1=50, t1=2, method='linear')

def compute_cwt(signal, fs, wavelet_name, scales=np.arange(1, 128)):
    coefs, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=1/fs)
    return np.abs(coefs)

wavelets = ['morl', 'cmor1.5-1.0', 'gaus4']
scales = np.arange(1, 128)

cwt_results = {w: compute_cwt(signal, fs, w, scales) for w in wavelets}

def scales_to_freqs(wavelet, scales, fs):
    freqs = pywt.scale2frequency(wavelet, scales) * fs
    return freqs

freq_axis = {w: scales_to_freqs(w, scales, fs) for w in wavelets}

for w in wavelets:
    plt.figure(figsize=(10, 4))
    plt.imshow(cwt_results[w], extent=[0, 2, freq_axis[w][-1], freq_axis[w][0]],
               cmap='jet', aspect='auto')
    plt.colorbar(label="Amplitude")
    plt.title(f"CWT Scalogram using {w}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


def ridge_from_cwt(cwt_matrix, freqs):
    ridge_freqs = []
    for col in range(cwt_matrix.shape[1]):
        idx = np.argmax(cwt_matrix[:, col])
        ridge_freqs.append(freqs[idx])
    return ridge_freqs

# Apply to Morlet (or try others)
ridge_freq = ridge_from_cwt(cwt_results['morl'], freq_axis['morl'])

# Plot with theoretical chirp
# plt.figure(figsize=(10, 4))
# plt.plot(t, ridge_freq, label="Detected Ridge (Instantaneous Frequency)")
# plt.plot(t, 5 + (45 * t / 2), label="True Chirp Frequency", linestyle='--')
# plt.title("Ridge Detection vs True Frequency")
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

fs = 512
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + 0.2*np.sin(2*np.pi*50*t)

# Create Wavelet Packet decomposition
wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=4)

# Extract all nodes at level 3
nodes = [node.path for node in wp.get_level(3, 'natural')]
print("Level 3 nodes:", nodes)



def shannon_entropy(x):
    x = np.asarray(x)
    p = np.square(x)
    p = p[p > 0]  # Avoid log(0)
    return -np.sum(p * np.log2(p))

# Calculate entropy for each node at deepest level
entropy_dict = {}
for node in wp.get_leaf_nodes():
    entropy_dict[node.path] = shannon_entropy(node.data)

# Select best basis (lowest entropy sum path)
best_basis = sorted(entropy_dict.items(), key=lambda item: item[1])
print("Best nodes by Shannon entropy:")
for node, val in best_basis[:5]:
    print(f"{node}: {val:.4f}")

coeffs = pywt.wavedec(signal, 'db4', level=4)
reconstructed_dwt = pywt.waverec(coeffs, 'db4')

# Wavelet Packet reconstruction using best basis nodes
wp_best = pywt.WaveletPacket(data=None, wavelet='db4', mode='symmetric')
for path, _ in best_basis[:8]:  # Use top 8 nodes
    wp_best[path] = wp[path].data
reconstructed_wp = wp_best.reconstruct(update=True)

min_len = min(len(t), len(reconstructed_wp))

# Plot original vs reconstructed
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Original', alpha=0.7)
plt.plot(t, reconstructed_dwt, label='DWT Reconstructed', linestyle='--')
plt.plot(t[:min_len], reconstructed_wp[:min_len], label='Wavelet Packet Reconstructed', linestyle=':')
plt.legend()
plt.title("Comparison of Original vs DWT vs Wavelet Packet")
plt.tight_layout()
plt.show()

