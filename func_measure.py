import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import threading
import time
from func_helper import *
import time
import sys

def suppress_animation_error(exctype, value, traceback):
    if exctype == AttributeError and "interval" in str(value):
        # Ignore this specific error
        return
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = suppress_animation_error

def noiseSoundcard(SETTINGS, MAX_AVERAGES, AUDIO_BUFFERSIZE, output_filename):
    SAMPLE_RATE, RESOLUTION, OVERLAP, _, _ = SETTINGS

    # Derived parameters
    WINDOW_SAMPLES = int(SAMPLE_RATE // RESOLUTION)
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)
    DISPLAY_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
    DISPLAY_SAMPLES = WINDOW_SAMPLES
    TOTAL_SAMPLES_NEEDED = WINDOW_SAMPLES + (MAX_AVERAGES - 1) * (WINDOW_SAMPLES - OVERLAP_SAMPLES)

    # Buffers
    audio_buffer = np.zeros((WINDOW_SAMPLES, 2))  # 2 channels
    recorded_data = []
    buffer_lock = threading.Lock()

    # Initialize plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    x_time = np.arange(DISPLAY_SAMPLES) / SAMPLE_RATE
    x_freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1 / SAMPLE_RATE)

    def initialize_plot():
        line1, = axs[0, 0].plot(x_time, np.zeros(DISPLAY_SAMPLES), label="Channel 1")
        line2, = axs[0, 0].plot(x_time, np.zeros(DISPLAY_SAMPLES), label="Channel 2")
        axs[0, 0].set_ylim(-1, 1)
        axs[0, 0].set_xlim(0, DISPLAY_SECONDS)
        axs[0, 0].legend()
        axs[0, 0].set_xlabel("Time [s]")
        axs[0, 0].set_ylabel("Amplitude")
        axs[0, 0].set_title("Live Audio Signal")

        spectrum1, = axs[0, 1].plot(x_freq, np.zeros(len(x_freq)), label="Channel 1")
        spectrum2, = axs[0, 1].plot(x_freq, np.zeros(len(x_freq)), label="Channel 2")
        axs[0, 1].set_ylim(-200, -50)
        axs[0, 1].set_xlim(200, SAMPLE_RATE / 2)
        axs[0, 1].set_xscale("log")
        axs[0, 1].legend()
        axs[0, 1].set_xlabel("Frequency [Hz]")
        axs[0, 1].set_ylabel("Magnitude [dB]")
        axs[0, 1].set_title("Live Spectra (Averaged)")

        transfer_func1, = axs[1, 1].plot(x_freq, np.zeros(len(x_freq)), label="H1")
        transfer_func2, = axs[1, 1].plot(x_freq, np.zeros(len(x_freq)), label="H2")
        axs[1, 1].set_ylim(-30, 30)
        axs[1, 1].set_xlim(200, SAMPLE_RATE / 2)
        axs[1, 1].set_xscale("log")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("Frequency [Hz]")
        axs[1, 1].set_ylabel("Magnitude [dB]")
        axs[1, 1].set_title("Transfer Function (Averaged)")

        coherence_line, = axs[1, 0].plot(x_freq, np.zeros(len(x_freq)), label="Coherence")
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_xlim(200, SAMPLE_RATE / 2)
        axs[1, 0].set_xscale("log")
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("Frequency [Hz]")
        axs[1, 0].set_ylabel("Coherence")
        axs[1, 0].set_title("Coherence (Averaged)")

        # Add a suptitle for the counter
        fig.suptitle(f"Samples: 0/{TOTAL_SAMPLES_NEEDED} (0/{MAX_AVERAGES} Averages)", fontsize=14, color="red")

        return fig, axs, line1, line2, spectrum1, spectrum2, transfer_func1, transfer_func2, coherence_line

    fig, axs, line1, line2, spectrum1, spectrum2, transfer_func1, transfer_func2, coherence_line = initialize_plot()

    Pxx_running_avg = None
    Pyy_running_avg = None
    Pxy_running_avg = None
    segment_count = 0

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal Pxx_running_avg, Pyy_running_avg, Pxy_running_avg, segment_count
        with buffer_lock:
            recorded_data_copy = np.concatenate(recorded_data, axis=0) if recorded_data else np.zeros((0, 2))
            total_samples = len(recorded_data_copy)
            line1.set_ydata(audio_buffer[:, 0])
            line2.set_ydata(audio_buffer[:, 1])

            while total_samples >= (segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES) + WINDOW_SAMPLES) and segment_count < MAX_AVERAGES:
                start_idx = segment_count * (WINDOW_SAMPLES - OVERLAP_SAMPLES)
                end_idx = start_idx + WINDOW_SAMPLES
                if end_idx > total_samples:
                    break
                segment_ch1 = recorded_data_copy[start_idx:end_idx, 0]
                segment_ch2 = recorded_data_copy[start_idx:end_idx, 1]

                f, Pxx = welch(segment_ch1, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, scaling='density')
                f, Pyy = welch(segment_ch2, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, scaling='density')
                f, Pxy = csd(segment_ch1, segment_ch2, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES, noverlap=0, scaling='density')

                if segment_count == 0:
                    Pxx_running_avg = Pxx
                    Pyy_running_avg = Pyy
                    Pxy_running_avg = Pxy
                else:
                    n = segment_count + 1
                    Pxx_running_avg = ((n - 1) * Pxx_running_avg + Pxx) / n
                    Pyy_running_avg = ((n - 1) * Pyy_running_avg + Pyy) / n
                    Pxy_running_avg = ((n - 1) * Pxy_running_avg + Pxy) / n

                segment_count += 1

            if segment_count > 0:
                H1 = Pxy_running_avg / Pxx_running_avg
                H2 = Pyy_running_avg / np.conj(Pxy_running_avg)

                coherence = np.abs(Pxy_running_avg)**2 / (Pxx_running_avg * Pyy_running_avg)
                coherence = np.clip(coherence, 0, 1)

                spectrum1.set_ydata(dB(Pxx_running_avg))
                spectrum2.set_ydata(dB(Pyy_running_avg))
                transfer_func1.set_ydata(dB(H1))
                transfer_func2.set_ydata(dB(H2))
                coherence_line.set_ydata(coherence)

            # Update the suptitle with the counter
            fig.suptitle(f"Samples: {total_samples}/{TOTAL_SAMPLES_NEEDED} ({segment_count}/{MAX_AVERAGES} Averages)", fontsize=14, color="red")

            if segment_count >= MAX_AVERAGES:
                # Save the recorded data to a CSV file
                recorded_data_array = np.concatenate(recorded_data, axis=0)[:TOTAL_SAMPLES_NEEDED]
                np.savetxt(output_filename, recorded_data_array, delimiter=",", header="Channel 1,Channel 2", comments="")
                print(f"Time-domain signals saved to '{output_filename}'.")
                plt.close(fig)
                return []

        return line1, line2, spectrum1, spectrum2, transfer_func1, transfer_func2, coherence_line

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, blocksize=AUDIO_BUFFERSIZE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()

def impactSoundcard(SETTINGS, IMPACT_THRESHOLD, IMPACT_COUNT, AUDIO_BUFFERSIZE, output_filename):
    timeout_seconds = 60
    start_time = time.time()

    SAMPLE_RATE, RESOLUTION, _, BEFORE, AFTER = SETTINGS

    # Parameters
    WINDOW_BEFORE = int(BEFORE * SAMPLE_RATE)
    WINDOW_AFTER = int(AFTER * SAMPLE_RATE)
    IMPACT_WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER
    FFT_SIZE = int(SAMPLE_RATE // RESOLUTION)

    # Buffer settings
    DISPLAY_SECONDS = 1
    DISPLAY_SAMPLES = DISPLAY_SECONDS * SAMPLE_RATE
    audio_buffer = np.zeros((DISPLAY_SAMPLES, 2))
    recorded_data = []
    impact_signals = []
    impact_count = 0
    last_impact_time = None

    buffer_lock = threading.Lock()

    # 2x3 subplot grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    x_time = np.arange(0, DISPLAY_SAMPLES) / SAMPLE_RATE
    x_freq = np.fft.rfftfreq(FFT_SIZE, d=1/SAMPLE_RATE)
    impact_time = np.linspace(-BEFORE, AFTER, IMPACT_WINDOW_SIZE)

#    # Top left: live audio
#    line1, = axs[0, 0].plot(x_time, np.zeros(DISPLAY_SAMPLES), label="Channel 1")
#    line2, = axs[0, 0].plot(x_time, np.zeros(DISPLAY_SAMPLES), label="Channel 2")
#    axs[0, 0].set_ylim(-1, 1)
#    axs[0, 0].set_xlim(0, DISPLAY_SECONDS)
#    axs[0, 0].legend()
#    axs[0, 0].set_xlabel("Time [s]")
#    axs[0, 0].set_ylabel("Amplitude")
#    axs[0, 0].set_title("Live Audio Signal (Waveform)")

    axs[0, 0].set_ylim(-5*IMPACT_THRESHOLD, 5*IMPACT_THRESHOLD)
    axs[0, 0].set_xlim(-BEFORE, AFTER)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].set_title("Force Impact Signals (Overlay)")
    axs[0, 0].legend(["Force"], loc="upper right")  

    # Top middle: superimposed spectra
    axs[0, 1].set_ylim(-150, -50)
    axs[0, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Magnitude [dB]")
    axs[0, 1].set_title("Impact Spectra (Overlay)")

    # Top right: averaged spectra
    spectrum1, = axs[0, 2].plot(x_freq, np.zeros(len(x_freq)), label="Channel 1")
    spectrum2, = axs[0, 2].plot(x_freq, np.zeros(len(x_freq)), label="Channel 2")
    axs[0, 2].set_ylim(-150, -50)
    axs[0, 2].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 2].set_xscale("log")
    axs[0, 2].legend()
    axs[0, 2].set_xlabel("Frequency [Hz]")
    axs[0, 2].set_ylabel("Magnitude [dB]")
    axs[0, 2].set_title("Averaged Spectra")

    # Bottom left: impact signals
    axs[1, 0].set_ylim(-5*IMPACT_THRESHOLD, 5*IMPACT_THRESHOLD)
    axs[1, 0].set_xlim(-BEFORE, AFTER)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].set_title("Response Impact Signals (Overlay)")
    impact_counter_text = axs[1, 0].text(
        0.5, 0.9, f"Impacts: {0}/{IMPACT_COUNT}",
        transform=axs[1, 0].transAxes, fontsize=12, color="red", ha="center"
    )

    # Bottom middle: superimposed transfer functions
    axs[1, 1].set_ylim(-40, 40)
    axs[1, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Magnitude [dB]")
    axs[1, 1].set_title("Transfer Functions (Overlay)")

    # Bottom right: averaged transfer function
    transfer_func, = axs[1, 2].plot(x_freq, np.zeros(len(x_freq)), label="H (Transfer Function)")
    axs[1, 2].set_ylim(-40, 40)
    axs[1, 2].set_xlim(200, SAMPLE_RATE / 2)
    axs[1, 2].set_xscale("log")
    axs[1, 2].legend()
    axs[1, 2].set_xlabel("Frequency [Hz]")
    axs[1, 2].set_ylabel("Magnitude [dB]")
    axs[1, 2].set_title("Averaged Transfer Function")

    spectra_ch1 = []
    spectra_ch2 = []
    transfer_ch12 = []

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal impact_count, last_impact_time
        
        if time.time() - start_time > timeout_seconds:
            print("Timeout reached. Closing measurement.")
            plt.close(fig)
            return

        with buffer_lock:
            #line1.set_ydata(audio_buffer[:, 0])
            #line2.set_ydata(audio_buffer[:, 1])

            if impact_count >= IMPACT_COUNT:
                recorded_data_array = np.concatenate(recorded_data, axis=0)
                np.savetxt(output_filename, recorded_data_array, delimiter=",", header="Channel 1,Channel 2", comments="")
                print(f"Time-domain signals saved to '{output_filename}'.")
                plt.close(fig)
                return

            current_time = time.time()
            if last_impact_time is None or (current_time - last_impact_time) >= DISPLAY_SECONDS:
                checked = abs(audio_buffer[WINDOW_BEFORE:-WINDOW_AFTER, 1])
                if np.max(checked) > IMPACT_THRESHOLD:
                    impact_index = next(a[0] for a in enumerate(checked) if a[1] > IMPACT_THRESHOLD)
                    start_index = impact_index
                    end_index = impact_index + IMPACT_WINDOW_SIZE
                    impact_signal = audio_buffer[start_index:end_index, :]
                    impact_signals.append(impact_signal)

                    padded_ch1 = np.pad(impact_signal[:, 0], (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode='constant')
                    padded_ch2 = np.pad(impact_signal[:, 1], (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode='constant')

                    XT = np.fft.rfft(padded_ch1) / FFT_SIZE
                    YT = np.fft.rfft(padded_ch2) / FFT_SIZE
                    H = YT / XT

                    spectra_ch1.append(XT)
                    spectra_ch2.append(YT)
                    transfer_ch12.append(H)

                    Pxx = np.abs(XT) ** 2
                    Pyy = np.abs(YT) ** 2
                    
                    Pxx_all = np.abs(spectra_ch1)**2
                    Pyy_all = np.abs(spectra_ch2)**2

                    # Overlay plots
                    axs[0, 1].plot(x_freq, 10*np.log10(Pxx), alpha=0.4, color='C0')
                    axs[0, 1].plot(x_freq, 10*np.log10(Pyy), alpha=0.4, color='C1')
                    axs[1, 1].plot(x_freq, dB(H), alpha=0.4, color='C2')

                    # Averaged plots
                    spectrum1.set_ydata(10*np.log10(np.mean(Pxx_all, axis=0)))
                    spectrum2.set_ydata(10*np.log10(np.mean(Pyy_all, axis=0)))
                    transfer_func.set_ydata(dB(np.mean(transfer_ch12, axis=0)))

                    axs[0, 0].plot(impact_time, impact_signal[:, 0], alpha=0.5, color='C0')
                    axs[1, 0].plot(impact_time, impact_signal[:, 1], alpha=0.7, label=f"Impact {impact_count + 1} (Ch2)")

                    impact_count += 1
                    impact_counter_text.set_text(f"Impacts: {impact_count}/{IMPACT_COUNT}")
                    last_impact_time = current_time

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

        return spectrum1, spectrum2, transfer_func

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, blocksize=AUDIO_BUFFERSIZE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()

def pendulumSoundcard(SETTINGS, IMPACT_THRESHOLD, IMPACT_COUNT, AUDIO_BUFFERSIZE, output_filename):
    timeout_seconds = 60
    start_time = time.time()

    SAMPLE_RATE, RESOLUTION, _, BEFORE, AFTER = SETTINGS

    WINDOW_BEFORE = int(BEFORE * SAMPLE_RATE)
    WINDOW_AFTER = int(AFTER * SAMPLE_RATE)
    IMPACT_WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER
    FFT_SIZE = int(SAMPLE_RATE // RESOLUTION)

    DISPLAY_SECONDS = 1
    DISPLAY_SAMPLES = DISPLAY_SECONDS * SAMPLE_RATE
    audio_buffer = np.zeros((DISPLAY_SAMPLES, 2))  # <-- 2 channels
    recorded_data = []
    impact_signals = []
    spectra = []
    impact_count = 0
    last_impact_time = None

    buffer_lock = threading.Lock()
    
    # Initialize plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    x_time = np.arange(0, DISPLAY_SAMPLES) / SAMPLE_RATE
    impact_time = np.linspace(-BEFORE, AFTER, IMPACT_WINDOW_SIZE)
    x_freq = np.fft.rfftfreq(FFT_SIZE, d=1/SAMPLE_RATE)

    # Top left: live response
    line, = axs[0, 0].plot(x_time, np.zeros(DISPLAY_SAMPLES), label="Response")
    axs[0, 0].set_ylim(-5*IMPACT_THRESHOLD, 5*IMPACT_THRESHOLD)
    axs[0, 0].set_xlim(0, DISPLAY_SECONDS)
    axs[0, 0].legend()
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].set_title("Live Response Signal")

    # Bottom left: impact windows
    axs[1, 0].set_ylim(-5*IMPACT_THRESHOLD, 5*IMPACT_THRESHOLD)
    axs[1, 0].set_xlim(-BEFORE, AFTER)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].set_title("Detected Impacts (Overlay)")
    impact_counter_text = axs[1, 0].text(
        0.5, 0.9, f"Impacts: {0}/{IMPACT_COUNT}",
        transform=axs[1, 0].transAxes, fontsize=12, color="red", ha="center"
    )

    # Top right: overlaid spectra
    axs[0, 1].set_ylim(-150, -50)
    axs[0, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Magnitude [dB]")
    axs[0, 1].set_title("Impact Spectra (Overlay)")

    # Bottom right: averaged spectrum
    avg_spectrum_line, = axs[1, 1].plot(x_freq, np.zeros_like(x_freq), color="C1", lw=2, label="Averaged Spectrum")
    axs[1, 1].set_ylim(-150, -50)
    axs[1, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Magnitude [dB]")
    axs[1, 1].set_title("Averaged Spectrum")
    axs[1, 1].legend()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        with buffer_lock:
            audio_buffer[:-frames] = audio_buffer[frames:]
            audio_buffer[-frames:] = indata
            recorded_data.append(indata.copy())

    def update(frame):
        nonlocal impact_count, last_impact_time
        
        if time.time() - start_time > timeout_seconds:
            print("Timeout reached. Closing measurement.")
            plt.close(fig)
            return
        
        with buffer_lock:
            # Use channel 2 for response
            line.set_ydata(audio_buffer[:, 1])

            if impact_count >= IMPACT_COUNT:
                recorded_data_array = np.concatenate(recorded_data, axis=0)
                # Save only channel 2 as "Response"
                np.savetxt(output_filename, recorded_data_array[:, 1], delimiter=",", header="Response", comments="")
                print(f"Time-domain signals saved to '{output_filename}'.")
                plt.close(fig)
                return

            current_time = time.time()
            if last_impact_time is None or (current_time - last_impact_time) >= DISPLAY_SECONDS:
                checked = np.abs(audio_buffer[WINDOW_BEFORE:-WINDOW_AFTER, 1])  # Channel 2
                if np.max(checked) > IMPACT_THRESHOLD:
                    impact_index = np.argmax(checked > IMPACT_THRESHOLD)
                    start_index = impact_index
                    end_index = impact_index + IMPACT_WINDOW_SIZE
                    impact_signal = audio_buffer[start_index:end_index, 1]  # Channel 2
                    if len(impact_signal) == IMPACT_WINDOW_SIZE:
                        impact_signals.append(impact_signal.copy())

                        axs[1, 0].plot(impact_time, impact_signal, alpha=0.7)

                        padded = np.pad(impact_signal, (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode='constant')
                        
                        YT = np.fft.rfft(padded) / FFT_SIZE
                        spectra.append(np.abs(YT))

                        axs[0, 1].plot(x_freq, dB(YT), alpha=0.4, color='C1')
                        avg_spectrum_line.set_ydata(dB(np.mean(spectra, axis=0)))

                        impact_count += 1
                        impact_counter_text.set_text(f"Impacts: {impact_count}/{IMPACT_COUNT}")
                        last_impact_time = current_time

                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

        return line, avg_spectrum_line

    stream = sd.InputStream(channels=2, samplerate=SAMPLE_RATE, blocksize=AUDIO_BUFFERSIZE, callback=audio_callback)
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

    with stream:
        plt.tight_layout()
        plt.show()
