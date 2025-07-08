import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import threading
from func_helper import *
import sys

def suppress_animation_error(exctype, value, traceback):
    if exctype == AttributeError and "interval" in str(value):
        # Ignore this specific error
        return
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = suppress_animation_error


def calibrateAccelerometer(accel_channel, SENS, GAIN, SETTINGS=[1600, 0.25, 0.95], averages=100):
    a_sens = SENS[0]
    a_gain = GAIN[0]
    SAMPLE_RATE, RESOLUTION, OVERLAP = SETTINGS
    WINDOW_SAMPLES = int(SAMPLE_RATE // RESOLUTION)
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)

    audio_buffer = np.zeros((WINDOW_SAMPLES, 2))
    recorded_data = []
    buffer_lock = threading.Lock()

    fig, ax = plt.subplots(figsize=(8, 4))
    x_freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1/SAMPLE_RATE)
    line, = ax.plot(x_freq, np.zeros_like(x_freq))
    ax.set_xlim(157, 160)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("RMS (m/s²)")
    ax.set_title("Live Accelerometer Autospectrum (157-160 Hz)")
    ax.grid(True)

    # Counter text in top left
    counter_text = fig.text(0.01, 0.97, f"Averages: 0/{averages}", fontsize=14, color="red", ha="left", va="top")

    segment_count = 0
    Pxx_running_avg = None

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal segment_count, Pxx_running_avg
        with buffer_lock:
            recorded_data_copy = np.concatenate(recorded_data, axis=0) if recorded_data else np.zeros((0, 2))
            total_samples = len(recorded_data_copy)
            while total_samples >= (segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES) + WINDOW_SAMPLES) and segment_count < averages:
                start_idx = segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES)
                end_idx = start_idx + WINDOW_SAMPLES
                if end_idx > total_samples:
                    break
                segment = recorded_data_copy[start_idx:end_idx, accel_channel]
                segment_phys = segment / (a_sens) * a_gain
                f, Pxx = welch(segment_phys, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                if segment_count == 0:
                    Pxx_running_avg = Pxx
                else:
                    n = segment_count + 1
                    Pxx_running_avg = ((n - 1) * Pxx_running_avg + Pxx) / n
                segment_count += 1

            if segment_count > 0:
                idx = np.logical_and(x_freq >= 157, x_freq <= 160)
                line.set_data(x_freq[idx], np.sqrt(Pxx_running_avg[idx]))
                ax.set_ylim(0, np.max(np.sqrt(Pxx_running_avg[idx])) * 1.2)

            # Update only the averages in the counter
            counter_text.set_text(f"Averages: {segment_count}/{averages}")

            if segment_count >= averages:
                idx = np.logical_and(x_freq >= 157, x_freq <= 160)
                max_val = np.max(Pxx_running_avg[idx])
                max_freq = x_freq[idx][np.argmax(Pxx_running_avg[idx])]
                measured_acc = np.sqrt(max_val)
                suggested_gain = a_gain * 10 / measured_acc
                print(f"Peak at {max_freq:.3f} Hz. Suggested gain to achieve 10 m/s²: {suggested_gain:.3f}")
                plt.close(fig)
                return []

        return line,

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()

def calibrateForcetransducer(force_channel, accel_channel, SENS, GAIN, MASS, SETTINGS=[3200, 0.5, 0.90], averages=100):
    a_sens = SENS[0]
    f_sens = SENS[1]
    a_gain = GAIN[0]
    f_gain = GAIN[1]
    SAMPLE_RATE, RESOLUTION, OVERLAP = SETTINGS
    WINDOW_SAMPLES = int(SAMPLE_RATE // RESOLUTION)
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)

    audio_buffer = np.zeros((WINDOW_SAMPLES, 2))
    recorded_data = []
    buffer_lock = threading.Lock()

    fig, ax = plt.subplots(figsize=(8, 4))
    x_freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1/SAMPLE_RATE)
    line, = ax.plot(x_freq, np.zeros_like(x_freq))
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("TF (grams)")
    ax.set_title("Live Force Transducer Transfer Function")
    ax.grid(True)

    # Counter text in top left
    counter_text = fig.text(0.01, 0.97, f"Averages: 0/{averages}", fontsize=14, color="red", ha="left", va="top")

    segment_count = 0
    H_running_avg = None

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal segment_count, H_running_avg
        with buffer_lock:
            recorded_data_copy = np.concatenate(recorded_data, axis=0) if recorded_data else np.zeros((0, 2))
            total_samples = len(recorded_data_copy)
            while total_samples >= (segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES) + WINDOW_SAMPLES) and segment_count < averages:
                start_idx = segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES)
                end_idx = start_idx + WINDOW_SAMPLES
                if end_idx > total_samples:
                    break
                force = recorded_data_copy[start_idx:end_idx, force_channel] / (f_sens) * f_gain  # N
                accel = recorded_data_copy[start_idx:end_idx, accel_channel] / (a_sens) * a_gain  # m/s²
                f, Pxx = welch(force, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                f, Pyy = welch(accel, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                f, Pxy = csd(force, accel, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                # Transfer function: H = Accel / Force, but we want Force/Accel = mass
                H = np.abs(Pxy) / Pxx  # This is |S_fa|/S_ff, units: (m/s²)/N = 1/kg
                H_mass = 1 / H         # Now units: kg
                if segment_count == 0:
                    H_running_avg = H_mass
                else:
                    n = segment_count + 1
                    H_running_avg = ((n - 1) * H_running_avg + H_mass) / n
                segment_count += 1

            if segment_count > 0:
                line.set_data(x_freq, H_running_avg * 1000)

            counter_text.set_text(f"Averages: {segment_count}/{averages}")

            if segment_count >= averages:
                idx_500 = np.argmin(np.abs(x_freq - 500))
                measured_mass = H_running_avg[idx_500]
                suggested_gain = f_gain * MASS / measured_mass
                print(f"TF at 800 Hz: {measured_mass:.3f} kg. Suggested gain for TF={MASS} kg: {suggested_gain:.3f}")
                plt.close(fig)
                return []

        return line,

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()
        
def calibrateLaser(laser_channel, accel_channel, SENS, GAIN, SETTINGS=[3200, 0.5, 0.90], averages=100):
    a_sens = SENS[0]
    l_sens = SENS[2]
    a_gain = GAIN[0]
    l_gain = GAIN[2]
    SAMPLE_RATE, RESOLUTION, OVERLAP = SETTINGS
    WINDOW_SAMPLES = int(SAMPLE_RATE // RESOLUTION)
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)

    audio_buffer = np.zeros((WINDOW_SAMPLES, 2))
    recorded_data = []
    buffer_lock = threading.Lock()

    fig, ax = plt.subplots(figsize=(8, 4))
    x_freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1/SAMPLE_RATE)
    line, = ax.plot(x_freq, np.zeros_like(x_freq))
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("TF (Accel/Velocity, 1/s)")
    ax.set_title("Live Laser Transfer Function (Goal: TF=1 at 500 Hz)")
    ax.grid(True)

    # Counter text in top left
    counter_text = fig.text(0.01, 0.97, f"Averages: 0/{averages}", fontsize=14, color="red", ha="left", va="top")

    segment_count = 0
    H_running_avg = None

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal segment_count, H_running_avg
        with buffer_lock:
            recorded_data_copy = np.concatenate(recorded_data, axis=0) if recorded_data else np.zeros((0, 2))
            total_samples = len(recorded_data_copy)
            while total_samples >= (segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES) + WINDOW_SAMPLES) and segment_count < averages:
                start_idx = segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES)
                end_idx = start_idx + WINDOW_SAMPLES
                if end_idx > total_samples:
                    break
                # Convert to physical units
                laser = recorded_data_copy[start_idx:end_idx, laser_channel] / (l_sens) * l_gain  # velocity [m/s]
                accel = recorded_data_copy[start_idx:end_idx, accel_channel] / (a_sens) * a_gain  # acceleration [m/s²]
                f, Pxx = welch(laser, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                f, Pxy = csd(laser, accel, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                omega = 2 * np.pi * x_freq
                omega[omega == 0] = 1e-12  # avoid division by zero
                # Transfer function: Accel / (Velocity * omega)
                H = np.abs(Pxy) / (Pxx * omega)
                if segment_count == 0:
                    H_running_avg = H
                else:
                    n = segment_count + 1
                    H_running_avg = ((n - 1) * H_running_avg + H) / n
                segment_count += 1

            if segment_count > 0:
                line.set_data(x_freq, H_running_avg)

            counter_text.set_text(f"Averages: {segment_count}/{averages}")

            if segment_count >= averages:
                idx_400 = np.argmin(np.abs(x_freq - 400))
                measured_tf = H_running_avg[idx_400]
                suggested_gain = l_gain * measured_tf  # Gain needed to make TF=1
                print(f"TF at 400 Hz: {measured_tf:.3f} (target: 1). Suggested gain for TF=1: {suggested_gain:.3f}")
                plt.close(fig)
                return []

        return line,

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()       
        
def calibrateMicrophone(mic_channel, SENS, GAIN, LVL, FREQ, SETTINGS=[3200, 0.5, 0.90], averages=100):
    """
    Calibrate a microphone to 94 dB SPL (1 Pa RMS) at 1 kHz.
    """
    m_sens = SENS  # V/Pa or mV/Pa
    m_gain = GAIN
    SAMPLE_RATE, RESOLUTION, OVERLAP = SETTINGS
    WINDOW_SAMPLES = int(SAMPLE_RATE // RESOLUTION)
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)

    audio_buffer = np.zeros((WINDOW_SAMPLES, 2))
    recorded_data = []
    buffer_lock = threading.Lock()

    fig, ax = plt.subplots(figsize=(8, 4))
    x_freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1/SAMPLE_RATE)
    line, = ax.plot(x_freq, np.zeros_like(x_freq))
    ax.set_xlim(0.95*FREQ, 1.05*FREQ)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("dB SPL")
    ax.set_title("Live Microphone Autospectrum")
    ax.grid(True)
    ax.axhline(LVL, color='red', linestyle='--', linewidth=2, label=f"Target {LVL} dB SPL")

    counter_text = fig.text(0.01, 0.97, f"Averages: 0/{averages}", fontsize=14, color="red", ha="left", va="top")

    segment_count = 0
    Pxx_running_avg = None

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal segment_count, Pxx_running_avg
        with buffer_lock:
            recorded_data_copy = np.concatenate(recorded_data, axis=0) if recorded_data else np.zeros((0, 2))
            total_samples = len(recorded_data_copy)
            while total_samples >= (segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES) + WINDOW_SAMPLES) and segment_count < averages:
                start_idx = segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES)
                end_idx = start_idx + WINDOW_SAMPLES
                if end_idx > total_samples:
                    break
                segment = recorded_data_copy[start_idx:end_idx, mic_channel]
                # Convert to Pascals (Pa)
                segment_phys = segment / m_sens * m_gain
                f, Pxx = welch(segment_phys, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, nfft=WINDOW_SAMPLES)
                if segment_count == 0:
                    Pxx_running_avg = Pxx
                else:
                    n = segment_count + 1
                    Pxx_running_avg = ((n - 1) * Pxx_running_avg + Pxx) / n
                segment_count += 1

            if segment_count > 0:
                idx = np.logical_and(x_freq >= FREQ*0.95, x_freq <= FREQ*1.05)
                # Convert to dB SPL
                rms_vals = np.sqrt(Pxx_running_avg[idx])
                db_spl_vals = 20 * np.log10(rms_vals / 20e-6 + 1e-20)  # add small value to avoid log(0)
                line.set_data(x_freq[idx], db_spl_vals)
                ax.set_ylim(np.min(db_spl_vals)-3, np.max(db_spl_vals)+3)
            counter_text.set_text(f"Averages: {segment_count}/{averages}")

            if segment_count >= averages:
                idx = np.logical_and(x_freq >= FREQ-5, x_freq <= FREQ+5)
                max_val = np.max(Pxx_running_avg[idx])
                max_freq = x_freq[idx][np.argmax(Pxx_running_avg[idx])]
                measured_pa = np.sqrt(max_val)
                # 94 dB SPL = 1 Pa RMS
                # Convert SPL to Pa
                target_pa = 20e-6 * 10**(LVL/20)
                suggested_gain = m_gain * target_pa / measured_pa
                print(f"Peak at {max_freq:.2f} Hz. Measured RMS: {measured_pa:.4f} Pa. Suggested gain for 1 Pa RMS (94 dB SPL): {suggested_gain:.3f}")
                plt.close(fig)
                return []

        return line,

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()