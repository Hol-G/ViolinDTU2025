#include <SPI.h>

// ADXL345 Register Definitions
#define ADXL345_DEVID         0x00  // Device ID
#define ADXL345_POWER_CTL     0x2D  // Power control register
#define ADXL345_DATA_FORMAT   0x31  // Data format control
#define ADXL345_DATAX0        0x32  // X-axis data 0 (first byte)
#define ADXL345_BW_RATE       0x2C  // Bandwidth rate register
#define ADXL345_INT_ENABLE    0x2E  // Interrupt enable control
#define ADXL345_INT_MAP       0x2F  // Interrupt mapping control
#define ADXL345_INT_SOURCE    0x30  // Interrupt source register

// SPI Pins for Arduino Nano
#define CS_PIN 10  // Chip Select pin connect to D10
#define INT1_PIN 2 // INT1 pin connected to D2

// ADXL345 Device ID
#define EXPECTED_ID 0xE5 

// Buffer Configuration
#define BUFFER_SIZE 128  // Number of samples per buffer
#define TOTAL_BUFFER_SIZE (BUFFER_SIZE * (2 + 4))  // 2 bytes Z + 4 bytes timestamp per sample

int16_t bufferA[BUFFER_SIZE];
int16_t bufferB[BUFFER_SIZE];
uint32_t bufferA_time[BUFFER_SIZE];
uint32_t bufferB_time[BUFFER_SIZE];

volatile int16_t* activeBuffer = bufferA;  // Points to the buffer being filled
volatile int16_t* transmitBuffer = nullptr;  // Points to the buffer to transmit
volatile uint32_t* activeBuffer_time = bufferA_time;
volatile uint32_t* transmitBuffer_time = nullptr;

volatile uint16_t bufferIndex = 0;  // Index in the active buffer
volatile bool bufferReady = false;  // Flag to indicate a buffer is ready to transmit
volatile uint16_t sampleCount = 0;  // Count samples to skip initial invalid ones

void setup() {
  Serial.begin(230400);  // Initialize Serial
  SPI.begin();
  SPI.setDataMode(SPI_MODE3);  // ADXL345 uses Mode 3
  SPI.setClockDivider(SPI_CLOCK_DIV8);  // 2 MHz SPI clock
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  pinMode(INT1_PIN, INPUT);  // Set D2 as input for INT1

  // Verify ADXL345
  if (readRegister(ADXL345_DEVID) != EXPECTED_ID) {
    Serial.println("ADXL345 not found!");
    while (true);  // Halt if device not found
  }

  Serial.println("ADXL345 initialized successfully");

  // Configure ADXL345
  writeRegister(ADXL345_DATA_FORMAT, 0x0B);  // ±16g range, full resolution
  writeRegister(ADXL345_BW_RATE, 0x0E);     // 3200 Hz data rate    
  writeRegister(ADXL345_INT_ENABLE, 0x00);  // Disable all interrupts first
  writeRegister(ADXL345_INT_MAP, 0x00);     // Map DATA_READY to INT1    
  writeRegister(ADXL345_POWER_CTL, 0x08);   // Measurement mode

  // Perform dummy reads to clear data registers
  for (uint8_t i = 0; i < 16; i++) {
    readZAxis();  // Discard results
  }

  // Enable DATA_READY interrupt
  writeRegister(ADXL345_INT_ENABLE, 0x80);

  // Clear any pending interrupts
  readRegister(ADXL345_INT_SOURCE);

  // Attach interrupt
  attachInterrupt(digitalPinToInterrupt(INT1_PIN), dataReadyInterrupt, RISING);
}

void loop() {
  if (bufferReady) {
    // Transmit Z and timestamp interleaved: [z0, t0, z1, t1, ...]
    for (uint16_t i = 0; i < BUFFER_SIZE; i++) {
      Serial.write((uint8_t*)&transmitBuffer[i], 2);      // Z value (int16_t)
      Serial.write((uint8_t*)&transmitBuffer_time[i], 4); // Timestamp (uint32_t)
    }
    bufferReady = false;  // Reset flag
  }
}

void dataReadyInterrupt() {
  // Read Z-axis
  int16_t z = readZAxis();
  uint32_t t = micros();
  sampleCount++;

  // Skip initial invalid samples (first 16 or out-of-range)
  if (sampleCount <= 16 || z < -4096 || z > 4096) {
    return;  // Discard invalid sample
  }

  // Store in active buffer
  activeBuffer[bufferIndex] = z;
  activeBuffer_time[bufferIndex] = t;
  bufferIndex++;

  // Check if buffer is full
  if (bufferIndex >= BUFFER_SIZE) {
    // Swap buffers
    transmitBuffer = activeBuffer;
    transmitBuffer_time = activeBuffer_time;
    activeBuffer = (activeBuffer == bufferA) ? bufferB : bufferA;
    activeBuffer_time = (activeBuffer_time == bufferA_time) ? bufferB_time : bufferA_time;
    bufferIndex = 0;
    bufferReady = true;  // Signal that a buffer is ready to transmit
  }
}

void writeRegister(uint8_t reg, uint8_t value) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(reg);
  SPI.transfer(value);
  digitalWrite(CS_PIN, HIGH);
}

uint8_t readRegister(uint8_t reg) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t value = SPI.transfer(0);
  digitalWrite(CS_PIN, HIGH);
  return value;
}

int16_t readZAxis() {
  uint8_t data[2];
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(ADXL345_DATAX0 + 4 | 0x80 | 0x40); // Start reading Z-axis, multi-byte read
  data[0] = SPI.transfer(0); // Read LSB
  data[1] = SPI.transfer(0); // Read MSB
  digitalWrite(CS_PIN, HIGH);
  return (int16_t)((data[1] << 8) | data[0]);
}