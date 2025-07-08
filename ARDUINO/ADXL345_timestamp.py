import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import struct

# Settings
PORT = "/dev/ttyUSB0"  # Adjust if needed
BAUD_RATE = 230400
SAMPLE_RATE = 1600
DURATION = 30  # seconds of data you want to keep after trimming
SAMPLES = SAMPLE_RATE * (DURATION + 2)  # +2 seconds for trimming

output_file = "ADXL_-1g 3.txt"

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Reading Z-axis data with timestamps (binary format, Ctrl+C to stop)...")

    buffer_z = np.zeros(SAMPLES)
    buffer_t = np.zeros(SAMPLES, dtype=np.uint32)
    buffer_index = 0

    with open(output_file, "w") as file:
        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < (DURATION + 2):
            try:
                # Each sample: 2 bytes Z + 4 bytes timestamp = 6 bytes
                if ser.in_waiting >= 6:
                    data_bytes = ser.read(6)
                    z = struct.unpack('<h', data_bytes[:2])[0]  # int16_t
                    t = struct.unpack('<I', data_bytes[2:])[0]  # uint32_t

                    buffer_z[buffer_index % SAMPLES] = z
                    buffer_t[buffer_index % SAMPLES] = t
                    buffer_index += 1

                    file.write(f"{z},{t}\n")
                    sample_count += 1

            except Exception as e:
                print(f"Error reading data: {e}")

        elapsed_time = time.time() - start_time
        sample_rate = sample_count / elapsed_time
        print(f"Recording complete. Data saved to arduino_z_data_with_time.txt.")
        print(f"Total samples recorded: {sample_count}")
        print(f"Effective sample rate: {sample_rate:.2f} Hz")

    # Convert timestamps from micros to seconds
    t_sec = (buffer_t[:buffer_index] - buffer_t[0]) * 1e-6
    z_vals = buffer_z[:buffer_index]

    # Discard first and last 1 second of data (before sorting)
    mask = (t_sec >= 1.0) & (t_sec <= (t_sec[-1] - 1.0))
    t_sec_trimmed = t_sec[mask]
    z_vals_trimmed = z_vals[mask]

    # Sort by time
    sort_idx = np.argsort(t_sec_trimmed)
    t_sec_sorted = t_sec_trimmed[sort_idx]
    z_vals_sorted = z_vals_trimmed[sort_idx]

    # Shift time to start at zero
    if len(t_sec_sorted) > 0:
        t_sec_sorted = t_sec_sorted - t_sec_sorted[0]

    # After shifting time to start at zero
    if len(t_sec_sorted) > 0:
        t_sec_sorted = t_sec_sorted - t_sec_sorted[0]

    # Check total timespan
    total_timespan = t_sec_sorted[-1] if len(t_sec_sorted) > 0 else 0
    expected_span = DURATION
    tolerance = 0.5  # seconds

    if abs(total_timespan - expected_span) > tolerance:
        print(f"Warning: Timespan after trimming is {total_timespan:.2f}s, expected ~{expected_span}s.")
        print("Check your data acquisition duration and trimming settings.")
    else:
        print(f"Timespan after trimming: {total_timespan:.2f}s (expected ~{expected_span}s)")


    # Compute normalized time gaps
    dt = np.diff(t_sec_sorted)
    intended_gap = 1.0 / SAMPLE_RATE
    dt_norm = dt / intended_gap

    fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})

    # Time signal
    axs[0].plot(t_sec_sorted, z_vals_sorted, label="Z-axis")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Z [raw]")
    axs[0].set_title("ADXL345 Z-Axis Acceleration Data (1s trimmed at start/end)")
    axs[0].legend()
    axs[0].grid()

    # Distribution of normalized time gaps
    print()
    
    if len(dt_norm) > 0:
        bins = np.arange(0, np.max(dt_norm) + 0.5, 0.1)
        axs[1].hist(dt_norm, bins=bins, color='gray', edgecolor='black')
    axs[1].set_xlabel("Time gap / Intended gap")
    axs[1].set_ylabel("Count")
    axs[1].set_title("Distribution of normalized time gaps between samples")
    axs[1].grid(axis='y')

    plt.tight_layout()
    plt.show()
    
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()