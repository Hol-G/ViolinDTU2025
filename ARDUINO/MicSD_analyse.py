import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sounddevice as sd

SAMPLE_RATE = 1600  # Hz, matches Arduino sketch
INPUT_FILE = 'audio_received.csv'
OUTPUT_FILE = 'audio_processed.csv'  # New file for processed signal

def load_csv_data(filename):
    samples = np.loadtxt(filename, delimiter=',')
    samples = samples - np.mean(samples)  # Remove DC offset
    last_half = len(samples) // 8
    samples = samples[last_half:-last_half]  # Use only the first half of the samples
    return samples

def perform_fft(samples):
    N = len(samples)
    samples_volts = samples * (3.3 / 1023.0)  # <-- 3.3V reference
    yf = fft(samples_volts)
    xf = fftfreq(N, 1 / SAMPLE_RATE)[:N//2]
    amplitudes = 2.0 / N * np.abs(yf[:N//2])
    return xf, amplitudes

def plot_results(samples):
    samples_volts = samples * (3.3 / 1023.0)  # <-- 3.3V reference
    timestamps = np.arange(len(samples)) / SAMPLE_RATE
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, samples_volts)
    plt.title('Time-Domain Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid()

    plt.subplot(2, 1, 2)
    xf, amplitudes = perform_fft(samples)
    plt.plot(xf, 20*np.log10(amplitudes))
    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.xlim(200, 800)  # Limit x-axis to 1600 Hz
    plt.title('Frequency Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
        
    plt.tight_layout()
    plt.show()
    
    
def play_audio(samples, sample_rate=SAMPLE_RATE):
    # Normalize to range [-1, 1] for sounddevice
    audio = (samples - 512) / 512.0  # Center and scale for Arduino 10-bit ADC
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

def main():
    samples = load_csv_data(INPUT_FILE)
    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")
    # Save processed samples
    samples_volts = samples * (3.3 / 1023.0)  # <-- 3.3V reference
    np.savetxt(OUTPUT_FILE, samples_volts, delimiter=',')
    print(f"Processed audio saved to {OUTPUT_FILE}")
    plot_results(samples)

if __name__ == "__main__":
    main()