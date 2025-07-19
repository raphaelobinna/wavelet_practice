import numpy as  np
import cv2
import matplotlib.pyplot as plt
from skimage import data, util, restoration
import pywt

def add_gaussian_noise(image):
    image = image.astype(np.float64) /255.0
    noisy_image = util.random_noise(image, mode='gaussian')

    return (noisy_image*255).astype(np.float64)


def compute_snr(original, next_original):
    original = original.astype(np.float64)
    next_original = next_original.astype(np.float64)

    signal_power = np.sum(original **2)
    noise_power = np.sum((original-next_original)**2)
    if noise_power ==0:
        return float('inf')
    return 10*np.log10(signal_power/noise_power)

def add_gaussian_noise_snr(image, target_snr_db=20):
    """Add noise to achieve specific SNR"""
    image = image.astype(np.float64)
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.float64)

def denoise_img(noisy_image, wavelet='db4'):
    noisy_image = noisy_image.astype(np.float64)

    noise_level_estimate = restoration.estimate_sigma(noisy_image, channel_axis=-1)

    denoised_channels = []
    for channel_index in range(noisy_image.shape[2]):
        channel = noisy_image[:, :, channel_index]
        sigma = noise_level_estimate[channel_index]
        threshold = 0.5 *sigma * np.sqrt(2* np.log(channel.size))

        coeffs = pywt.wavedec2(channel,wavelet, level=2)

        coeffs_threshold = coeffs.copy()

        for i in range(1,len(coeffs_threshold)):
            coeffs_threshold[i] = tuple(pywt.threshold(subband,threshold, mode='soft') for subband in coeffs[i])

        reconstructed_channel = pywt.waverec2(coeffs_threshold, wavelet)
        reconstructed_channel = np.clip(reconstructed_channel,0,255)
        denoised_channels.append(reconstructed_channel)

    denoised_img = cv2.merge([ch.astype(noisy_image.dtype) for ch in denoised_channels])
    

    return denoised_img

def denoise_wavelet_skimage(noisy_image, wavelet='db4', method='BayesShrink'):
    noisy_image_norm = noisy_image / 255.0  # must be in [0, 1] for skimage
    denoised = restoration.denoise_wavelet(
        noisy_image_norm,
        channel_axis=-1,
        wavelet=wavelet,
        method=method,
        mode='soft',
        rescale_sigma=True
    )
    return (denoised * 255).astype(np.uint8)

def denoise_for_each(noisy_image, wavelet='sym8'):
    # Ensure image is uint8 or float64 [0, 255]
    if noisy_image.max() <= 1.0:
        noisy_image = (noisy_image * 255).astype(np.float64)
    else:
        noisy_image = noisy_image.astype(np.float64)

    channels = cv2.split(noisy_image)
    denoised_channels = []



    for channel in channels:
        # Wavelet decomposition
        coeffs = pywt.wavedec2(channel, wavelet, level=1)
        

        cA = coeffs[0]
        thresholded_details =[]

        for (cH, cV, cD) in coeffs[1:]:

            # Estimate noise via MAD
            sigma_h = np.median(np.abs(cH)) / 0.6745
            sigma_v = np.median(np.abs(cV)) / 0.6745
            sigma_d = np.median(np.abs(cD)) / 0.6745

            # Compute adaptive thresholds
            thresh_h = sigma_h * np.sqrt(2 * np.log(cH.size))
            thresh_v = sigma_v * np.sqrt(2 * np.log(cV.size))
            thresh_d = sigma_d * np.sqrt(2 * np.log(cD.size))

            # Apply soft thresholding
            cH_thresh = pywt.threshold(cH, thresh_h, mode='soft')
            cV_thresh = pywt.threshold(cV, thresh_v, mode='soft')
            cD_thresh = pywt.threshold(cD, thresh_d, mode='soft')

            thresholded_details.append((cH_thresh, cV_thresh, cD_thresh))

        # Reconstruct the channel
        coeffs_thresh = [cA] + thresholded_details
        denoised_channel = pywt.waverec2(coeffs_thresh, wavelet)
        denoised_channel = np.clip(denoised_channel, 0, 255)
        denoised_channels.append(denoised_channel)

    # Merge channels
    denoised_img = cv2.merge([ch.astype(np.uint8) for ch in denoised_channels])
    return denoised_img




# def adaptive_wavelet_denoise_db2(noisy_image, wavelet='db4', levels=2):
#     """
#     Advanced denoising with adaptive thresholding per subband
#     """
#     img_float = noisy_image.astype(np.float64)

#     max_level = pywt.dwt_max_level(min(img_float.shape[:2]), pywt.Wavelet(wavelet).dec_len)
#     levels = min(levels, max_level)
    
#     # Multi-level decomposition
#     coeffs = pywt.wavedec2(img_float, wavelet, mode='symmetric', level=levels)
    
#     # Adaptive thresholding for each level
#     coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    
#     for i in range(1, len(coeffs)):
#         # Calculate threshold for each subband separately
#         cH, cV, cD = coeffs[i]  # Horizontal, Vertical, Diagonal
        
#         # Estimate noise in each subband using median absolute deviation
#         sigma_h = np.median(np.abs(cH)) / 0.6745
#         sigma_v = np.median(np.abs(cV)) / 0.6745
#         sigma_d = np.median(np.abs(cD)) / 0.6745
        
#         # Calculate adaptive thresholds
#         thresh_h = sigma_h * np.sqrt(2 * np.log(cH.size))
#         thresh_v = sigma_v * np.sqrt(2 * np.log(cV.size))
#         thresh_d = sigma_d * np.sqrt(2 * np.log(cD.size))
        
#         # Apply soft thresholding
#         cH_thresh = pywt.threshold(cH, thresh_h, mode='soft')
#         cV_thresh = pywt.threshold(cV, thresh_v, mode='soft')
#         cD_thresh = pywt.threshold(cD, thresh_d, mode='soft')
        
#         coeffs_thresh.append((cH_thresh, cV_thresh, cD_thresh))
    
#     # Reconstruct
#     denoised = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
#     denoised = np.clip(denoised, 0, 255).astype(noisy_image.dtype)

#     if denoised.shape[2] == 4:
#         denoised = denoised[:, :, :3]
    
#     return denoised


if __name__ == "__main__":
    new_image = cv2.imread('xray.png')

    

    noisy_image = add_gaussian_noise_snr(new_image)

    
    denoised_image =denoise_img(noisy_image)
    sk_deenoise = denoise_wavelet_skimage(noisy_image)
    denois_each =denoise_for_each(noisy_image)

    snr_before = compute_snr(new_image, noisy_image)
    snr_after = compute_snr(new_image, denoised_image)
    snr_sk = compute_snr(new_image, sk_deenoise)
    snr_each =compute_snr(new_image, denois_each)

    print(f"SNR before denoising: {snr_before:.2f} dB")
    print(f"SNR after denoising: {snr_after:.2f} dB")
    print(f"SNR after SK: {snr_sk:.2f} dB")
    print(f"SNR after Each: {snr_each:.2f} dB")

    cv2.imshow("Original", new_image)
    cv2.imshow("Noisy", noisy_image.astype(np.uint8))
    cv2.imshow("Denoised", denoised_image.astype(np.uint8))
    cv2.imshow("sk", sk_deenoise)
    cv2.imshow("Each", denois_each)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

