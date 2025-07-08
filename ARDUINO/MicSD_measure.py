import serial
import time

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

def start_recording():
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize

    # Send start command
    print("Sending START command...")
    ser.write(b'START\n')

    # Wait for "Recording finished." message
    print("Waiting for 'Recording finished.' from Arduino...")
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"~ {line}")
            if "Recording finished." in line:
                break
        time.sleep(0.1)
    ser.close()
    print("Recording finished. Ready to read data.")

if __name__ == "__main__":
    start_recording()
