import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 3200  # Hz, matches Arduino sketch
INPUT_FILE = 'audio_received.csv'

def load_csv_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    timestamps = data[:, 0] / 1e6  # Convert microseconds to seconds
    samples = data[:, 1]
    samples = samples - np.mean(samples)  # Remove DC offset
    return timestamps, samples

def perform_fft(samples):
    N = len(samples)
    samples_volts = samples * (5.0 / 1023.0)
    yf = fft(samples_volts)
    xf = fftfreq(N, 1 / SAMPLE_RATE)[:N//2]
    amplitudes = 2.0 / N * np.abs(yf[:N//2])
    return xf, amplitudes

def plot_results(timestamps, samples):
    samples_volts = samples * (5.0 / 1023.0)
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, samples_volts)
    plt.title('Time-Domain Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid()

    plt.subplot(3, 1, 2)
    xf, amplitudes = perform_fft(samples)
    plt.plot(xf, amplitudes)
    plt.title('Frequency Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    # Compute time intervals between samples
    dt = np.diff(timestamps)

    # Plot histogram/bar chart of time intervals
    plt.subplot(3, 1, 3)
    plt.hist(dt*1e6 / 312.5, bins=1000, color='skyblue', edgecolor='k')
    plt.yscale('log')
    plt.xlabel("Time interval between samples [us]")
    plt.ylabel("Count")
    plt.title("Distribution of Time Intervals Between Samples")
    plt.grid()

    
    plt.tight_layout()
    plt.show()
    
    

def main():
    timestamps, samples = load_csv_data(INPUT_FILE)
    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")
    plot_results(timestamps, samples)

if __name__ == "__main__":
    main()