import numpy as np
from scipy.signal import find_peaks, savgol_filter

def dB(x): return 20 * np.log10(np.abs(x))
def pdB(x): return 10 * np.log10(np.abs(x))

def getFinalMeanStd(H_means, Fest_avg, Fest_std):
    H_means = np.stack(H_means)
    H_mean = np.mean(H_means, axis=0)
    rel_var_H = np.var(H_means, axis=0, ddof=1) / (H_mean**2)
    rel_var_Fest = (Fest_std / Fest_avg)**2
    rel_var_comb = rel_var_Fest + 0.1 * rel_var_H
    std_comb = H_mean * np.sqrt(rel_var_comb)
    return H_mean, std_comb

def getHestAndStd(Y_means, Fest_avg, Fest_std):
    Y_means = np.stack(Y_means)
    Y_mean = np.mean(Y_means, axis=0)
    rel_var_Y = np.var(Y_means, axis=0, ddof=1) / (Y_mean**2)
    rel_var_Fest = (Fest_std / Fest_avg)**2
    rel_var_Hest = rel_var_Fest + 0.1 * rel_var_Y
    H_est = Y_mean / Fest_avg
    H_est_std = H_est * np.sqrt(rel_var_Hest)
    return H_est, H_est_std


def smooth(Xest):
    window_length = 201  # Must be odd, adjust for smoothness
    polyorder = 1

    # Only from 200 Hz onwards, meaning index 400
    Xest_low = Xest[:400]
    Xest = Xest[400:]

    Xest_dB = dB(Xest)
    Xest_dB_smooth = savgol_filter(Xest_dB, window_length, polyorder)
    Xest_dB_smooth = savgol_filter(Xest_dB_smooth, window_length, polyorder)
    Xest_dB_smooth = savgol_filter(Xest_dB_smooth, window_length, polyorder)

    Xest_mag_smooth = 10 ** (Xest_dB_smooth / 20)

    Xest_mag_smooth = np.concatenate((Xest_low, Xest_mag_smooth))

    return Xest_mag_smooth

def normalize(mean, std, freq, A0):
    index = np.argmin(np.abs(freq - A0))
    factor = np.abs(mean[index])
    mean = mean / factor
    std = std / factor
    return mean, std

def lowerAndUpper(mean, std):
    lower = np.abs(mean) - std
    upper = np.abs(mean) + std
    return lower, upper 

def getMeanStd(H):
    mean_H = np.mean(H, axis=0)
    std_H = np.std(H, axis=0, ddof=1)
    return mean_H, std_H

def accelToVel(f, H):
    omega = 2 * np.pi * f
    omega[0] = 1e-10
    H_vel = H / (1j * omega)
    return H_vel

def massCorrect(f, H, mass):
    omega = 2 * np.pi * f
    H_corrected = H / (1 - 1j*omega*mass*H)
    return H_corrected

def computeTransferCoherence(Pxx, Pyy, Pxy):
    h1 = Pxy / Pxx
    h2 = Pyy / np.conj(Pxy)
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)
    coh = np.clip(coh, 0, 1)
    return h1, h2, coh

def getPeaks(f, H, peak_ranges, prominences):
    peak_freq = []
    peak_mag = []

    for i in range(H.shape[0]):
        peaks, _ = find_peaks(np.abs(H[i]), prominence=prominences[i])
        peak_freq_i = []
        peak_mag_i = []

        for j in range(len(peak_ranges)):
            # Find peaks within the specified frequency range
            range_mask = (f >= peak_ranges[j][0]) & (f <= peak_ranges[j][1])
            range_freq = f[range_mask]
            range_mag = np.abs(H[i][range_mask])

            # Find peaks in the specified range
            peaks_in_range = peaks[(peaks >= peak_ranges[j][0]) & (peaks <= peak_ranges[j][1])]
            if len(peaks_in_range) > 0:
                max_peak_idx = peaks_in_range[np.argmax(range_mag[peaks_in_range])]
                peak_freq_i.append(range_freq[max_peak_idx])
                peak_mag_i.append(range_mag[max_peak_idx])

        peak_freq.append(peak_freq_i)
        peak_mag.append(peak_mag_i)
        
    return peak_freq, peak_mag

def normalizeFRF(A0, f, h):
    norm_idx = np.argmin(np.abs(f - A0))    
    h_normalized = h / np.abs(h[norm_idx])    
    return h_normalized

