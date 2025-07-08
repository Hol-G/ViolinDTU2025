import serial
import time
import csv
import numpy as np

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
OUTPUT_FILE = 'audio_received.csv'  # Local file to save data

def read_audio_data():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize

    print("Sending READ command...")
    ser.write(b'READ\n')

    print("Receiving binary data...")
    start_time = time.time()
    raw_bytes = bytearray()
    transfer_complete = False

    while time.time() - start_time < 120:  # Allow up to 2 minutes for transfer
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            raw_bytes.extend(chunk)
            if b"Data transfer complete" in raw_bytes:
                # Find the marker and cut off everything after it
                marker_index = raw_bytes.find(b"Data transfer complete")
                data_bytes = raw_bytes[:marker_index]
                transfer_complete = True
                break
        time.sleep(0.01)
    ser.close()

    if not transfer_complete:
        print("WARNING: Timeout or incomplete transfer.")
        data_bytes = raw_bytes

    # Remove any non-binary trailing text (e.g., newlines before/after marker)
    # Ensure even number of bytes for uint16_t
    if len(data_bytes) % 2 != 0:
        data_bytes = data_bytes[:-1]

    # Convert to numpy array of uint16
    samples = np.frombuffer(data_bytes, dtype='>u2')  # Little-endian
    samples = samples[20:]
    
    # Write to CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        for s in samples:
            writer.writerow([s])
    print(f"Data saved to {OUTPUT_FILE} ({len(samples)} samples)")

if __name__ == "__main__":
    read_audio_data()