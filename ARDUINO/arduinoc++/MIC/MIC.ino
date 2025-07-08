// Arduino sketch to read MAX4466 microphone and send data over serial
const int micPin = A0; // Microphone OUT connected to A0
const int sampleRate = 8000; // Sampling rate in Hz (8 kHz)
const unsigned long sampleInterval = 1000000 / sampleRate; // Microseconds per sample

void setup() {
  // Initialize serial communication at 115200 baud
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for Nano)
  }
  pinMode(micPin, INPUT);
}

void loop() {
  static unsigned long lastSampleTime = 0;
  unsigned long currentTime = micros();

  // Sample at the specified rate
  if (currentTime - lastSampleTime >= sampleInterval) {
    int micValue = analogRead(micPin); // Read analog value (0-1023)
    Serial.println(micValue); // Send value to serial port
    lastSampleTime = currentTime;
  }
}