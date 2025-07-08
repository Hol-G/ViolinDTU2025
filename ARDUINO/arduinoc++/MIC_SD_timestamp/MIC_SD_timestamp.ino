#include <SD.h>
#include <SPI.h>

const int micPin = A0;
const int chipSelect = 10;
const int sampleRate = 3200;
const int blockSize = 32;

volatile uint16_t bufferA[blockSize];
volatile uint16_t bufferB[blockSize];
volatile uint16_t bufferC[blockSize];
volatile uint32_t timestampsA[blockSize];
volatile uint32_t timestampsB[blockSize];
volatile uint32_t timestampsC[blockSize];

volatile bool bufferA_full = false;
volatile bool bufferB_full = false;
volatile bool bufferC_full = false;


volatile int bufferIndex = 0;
volatile bool recording = false;

File dataFile;

void setupTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A = 1249; // (16,000,000 / (8 * 3200)) - 1 = 624 for 3200 Hz
  TCCR1B |= (1 << WGM12); // CTC mode
  TCCR1B |= (1 << CS11);  // Prescaler 8
  TIMSK1 |= (1 << OCIE1A); // Enable compare match interrupt
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  if (!recording) return;
  if (currentBuffer == 0) {
    bufferA[bufferIndex] = analogRead(micPin);
    timestampsA[bufferIndex] = micros();
  } else if (currentBuffer == 1) {
    bufferB[bufferIndex] = analogRead(micPin);
    timestampsB[bufferIndex] = micros();
  } else {
    bufferC[bufferIndex] = analogRead(micPin);
    timestampsC[bufferIndex] = micros();
  }
  bufferIndex++;
  if (bufferIndex >= blockSize) {
    if (currentBuffer == 0) bufferA_full = true;
    else if (currentBuffer == 1) bufferB_full = true;
    else bufferC_full = true;
    currentBuffer = (currentBuffer + 1) % 3;
    bufferIndex = 0;
  }
}

void setup() {
  Serial.begin(2000000);
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
  if (SD.exists("audio.csv")) SD.remove("audio.csv");
  dataFile = SD.open("audio.csv", FILE_WRITE);
  if (!dataFile) {
    Serial.println("Error opening audio.csv");
    return;
  }
  recording = true;
  bufferA_full = false;
  bufferB_full = false;
  bufferC_full = false;
  bufferIndex = 0;
  currentBuffer = 0;
  unsigned long startMillis = millis();
  Serial.println("Recording started...");
  while (millis() - startMillis < 5000) { // 5 seconds
    if (bufferA_full) {
      noInterrupts();
      bufferA_full = false;
      interrupts();
      for (int i = 0; i < blockSize; i++) {
        dataFile.print(timestampsA[i]);
        dataFile.print(",");
        dataFile.println(bufferA[i]);
      }
    }
    if (bufferB_full) {
      noInterrupts();
      bufferB_full = false;
      interrupts();
      for (int i = 0; i < blockSize; i++) {
        dataFile.print(timestampsB[i]);
        dataFile.print(",");
        dataFile.println(bufferB[i]);
      }
    }
    if (bufferC_full) {
      noInterrupts();
      bufferC_full = false;
      interrupts();
      for (int i = 0; i < blockSize; i++) {
        dataFile.print(timestampsC[i]);
        dataFile.print(",");
        dataFile.println(bufferC[i]);
      }
    }
  }
  recording = false;
  // Write any remaining samples in the current buffer
  int remaining = bufferIndex;
  if (remaining > 0) {
    if (currentBuffer == 0) {
      for (int i = 0; i < remaining; i++) {
        dataFile.print(timestampsA[i]);
        dataFile.print(",");
        dataFile.println(bufferA[i]);
      }
    } else if (currentBuffer == 1) {
      for (int i = 0; i < remaining; i++) {
        dataFile.print(timestampsB[i]);
        dataFile.print(",");
        dataFile.println(bufferB[i]);
      }
    } else {
      for (int i = 0; i < remaining; i++) {
        dataFile.print(timestampsC[i]);
        dataFile.print(",");
        dataFile.println(bufferC[i]);
      }
    }
  }
  dataFile.close();
  Serial.println("Recording finished.");
}

void readAudioData() {
  dataFile = SD.open("audio.csv", FILE_READ);
  if (!dataFile) {
    Serial.println("Error opening audio.csv for reading");
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