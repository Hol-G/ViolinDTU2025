#include <SD.h>
#include <SPI.h>

const int micPin = A0;
const int chipSelect = 10;
const int sampleRate = 3200;
const int blockSize = 128;

volatile uint16_t bufferA[blockSize];
volatile uint16_t bufferB[blockSize];

volatile bool bufferA_full = false;
volatile bool bufferB_full = false;
volatile int bufferIndex = 0;
volatile int currentBuffer = 0; // 0: A, 1: B
volatile bool recording = false;

File dataFile;

void setupTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A = 624; // (16,000,000 / (8 * 3200)) - 1 = 624 for 3200 Hz
  TCCR1B |= (1 << WGM12); // CTC mode
  TCCR1B |= (1 << CS11);  // Prescaler 8
  TIMSK1 |= (1 << OCIE1A); // Enable compare match interrupt
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  if (!recording) return;
  if (currentBuffer == 0) {
    bufferA[bufferIndex] = analogRead(micPin);
  } else {
    bufferB[bufferIndex] = analogRead(micPin);
  }
  bufferIndex++;
  if (bufferIndex >= blockSize) {
    if (currentBuffer == 0) bufferA_full = true;
    else bufferB_full = true;
    currentBuffer = 1 - currentBuffer; // Toggle between 0 and 1
    bufferIndex = 0;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed!");
    while (1);
  }
  Serial.println("SD card initialized.");
  setupTimer1();
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "START") {
      recordAudio();
    } else if (command == "READ") {
      readAudioData();
    }
  }
}

void recordAudio() {
  if (SD.exists("audio.bin")) SD.remove("audio.bin");
  dataFile = SD.open("audio.bin", FILE_WRITE);
  if (!dataFile) {
    Serial.println("Error opening audio.bin");
    return;
  }
  recording = true;
  bufferA_full = false;
  bufferB_full = false;
  bufferIndex = 0;
  currentBuffer = 0;
  unsigned long startMillis = millis();
  Serial.println("Recording started...");
  while (millis() - startMillis < 30000) { // 5 seconds
    if (bufferA_full) {
      noInterrupts();
      bufferA_full = false;
      interrupts();
      dataFile.write((const uint8_t*)bufferA, blockSize * sizeof(uint16_t));
    }
    if (bufferB_full) {
      noInterrupts();
      bufferB_full = false;
      interrupts();
      dataFile.write((const uint8_t*)bufferB, blockSize * sizeof(uint16_t));
    }
  }
  recording = false;
  // Write any remaining samples in the current buffer
  int remaining = bufferIndex;
  if (remaining > 0) {
    if (currentBuffer == 0) {
      dataFile.write((const uint8_t*)bufferA, remaining * sizeof(uint16_t));
    } else {
      dataFile.write((const uint8_t*)bufferB, remaining * sizeof(uint16_t));
    }
  }
  dataFile.close();
  Serial.println("Recording finished.");
}

void readAudioData() {
  dataFile = SD.open("audio.bin", FILE_READ);
  if (!dataFile) {
    Serial.println("Error opening audio.bin for reading");
    return;
  }
  Serial.println("Sending data...");
  while (dataFile.available()) {
    char c = dataFile.read();
    Serial.write(c);
  }
  dataFile.close();
  Serial.println("Data transfer complete.");
}