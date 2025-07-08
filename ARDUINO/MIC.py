import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Replace with your Arduino's serial port (e.g., '/dev/ttyUSB0' on Linux)
BAUD_RATE = 115200
SAMPLE_RATE = 8000  # Matches Arduino sketch
DURATION = 2  # Seconds to collect data
NUM_SAMPLES = SAMPLE_RATE * DURATION

def collect_audio_data():
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize

    # Collect samples
    samples = []
    print("Collecting audio data...")
    start_time = time.time()
    while len(samples) < NUM_SAMPLES:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                sample = int(line)  # Convert serial data to integer
                samples.append(sample)
        except (ValueError, UnicodeDecodeError):
            continue  # Skip invalid data
    ser.close()

    # Convert to numpy array and scale to volts (0-5V for Arduino ADC)
    samples = np.array(samples[:NUM_SAMPLES])
    samples = samples * (5.0 / 1023.0)  # Scale to volts
    samples = samples - np.mean(samples)  # Remove DC offset
    samples = samples / np.max(np.abs(samples))  # Normalize to -1 to 1
    return samples

def perform_fft(samples):
    # Perform FFT
    N = len(samples)
    yf = fft(samples)
    xf = fftfreq(N, 1 / SAMPLE_RATE)[:N//2]  # Frequency bins (positive frequencies only)
    amplitudes = 2.0 / N * np.abs(yf[:N//2])  # Amplitude spectrum
    return xf, amplitudes

def plot_results(t, samples, xf, amplitudes):
    # Plot time-domain signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, samples)
    plt.title('Time-Domain Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid()

    # Plot frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(xf, amplitudes)
    plt.title('Frequency Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xscale('log')  # Log scale for frequency axis
    plt.xlim(200, SAMPLE_RATE / 2)  # Limit x-axis to Nyquist frequency
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    # Collect audio data
    samples = collect_audio_data()
    
    # Generate time axis
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    
    # Perform FFT
    xf, amplitudes = perform_fft(samples)
    
    # Plot results
    plot_results(t, samples, xf, amplitudes)

if __name__ == "__main__":
    main()