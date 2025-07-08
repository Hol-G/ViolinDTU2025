import serial
import time
import csv

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 2000000
OUTPUT_FILE = 'audio_received.csv'  # Local file to save data

def read_audio_data():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize

    print("Sending READ command...")
    ser.write(b'READ\n')

    timestamps = []
    samples = []
    print("Receiving data...")
    start_time = time.time()
    buffer = ""
    transfer_complete = False

    while time.time() - start_time < 120:  # Allow up to 2 minutes for transfer
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            lines = buffer.split('\n')
            buffer = lines[-1]
            for line in lines[:-1]:
                if ',' in line:
                    try:
                        timestamp, sample = line.split(',')
                        timestamps.append(float(timestamp) / 1000000.0)
                        samples.append(int(sample))
                    except (ValueError, IndexError):
                        continue
                if "Data transfer complete" in line:
                    transfer_complete = True
                    break
            if transfer_complete:
                break
        time.sleep(0.01)
    ser.close()

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        for t, s in zip(timestamps, samples):
            writer.writerow([t * 1000000, s])  # Save timestamps in microseconds
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    read_audio_data()