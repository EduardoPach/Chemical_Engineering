// Arduino Nano
// SDA - A4; SCL - A5
// Digital Pins for Interrupt: 2;3

#include <Wire.h>

int HzPerLiterInlet = 86;
int HzPerLiterOutlet = 33;
int interval = 3000;

// Sensors' pins
const byte pinInletFlow = 2;                                        // Pin to Interrupt first flow meter (pure water) YF-s401.
const byte pinOutletFlow = 3;                                       // Pin to Interrupt second flow meter (Air-Water mixture) DWS-MH-01.
int pinPiezo1 = A0;                                                 // Piezoeletric Ceramic Vibration sensor pin.
int pinPiezo2 = A1;                                                 // Piezoeltric Vibration sensor pin.
int pin801s = A2;                                                   // 801s Vibration sensor pin.
int pinPressure = A3;                                               // Differencial Pressure sensor pin.
int pinCond = A7;                                                   // Condutance Sensor pin (embedded in DWS-MH-01).

// Pulse variables
volatile int InletPulses;                                        // Inlet flow pulses to count - volatile because it's inside ISR (Interrupt Service Routines).
volatile int OutletPulses;                                       // Outlet flow pulses to count - volatile because it's inside ISR (Interrupt Service Routines).

// Storing Variables
double valueInletFlow;                                              // Inlet water flow - L/min.
double valueOutletFlow;                                             // Outlet Air-Water flow - L/min.
float valuePressure;                                                // Differencial Pressure - kPa.
int valueCond;                                                      // Condutance value.
int valuePiezo1;                                                    // Vibration value from piezo 1.
int valuePiezo2;                                                    // Vibration value from piezo2.
int value801s;                                                      // Vibration value from 801s.

// I2C Adress for MPU-6050 sensor
const int MPU = 0x68;

// Variables for MPU
float AccX, AccY, AccZ, Temp, GyrX, GyrY, GyrZ;

void setup() {
  // Begin Serial Transmission.
  Serial.begin(9600);  //baudrate
  // Setting MPU 6050 up.
  setMPU();
  // Attaching functions to interrupt pins for flow meters
  attachInterrupt(digitalPinToInterrupt(pinInletFlow),CountInletPulses, RISING);
  attachInterrupt(digitalPinToInterrupt(pinOutletFlow),CountOutletPulses, RISING);
  // Printing headers.
  Serial.print("Piezo1");Serial.print(",");
  Serial.print("Piezo2");Serial.print(",");
  Serial.print("801s");Serial.print(",");
  Serial.print("Condutance");Serial.print(",");
  Serial.print("dP");Serial.print(",");
  Serial.print("InletFlow");Serial.print(",");
  Serial.print("OutletFlow");Serial.print(",");
  Serial.print("AccX");Serial.print(",");
  Serial.print("AccY");Serial.print(",");
  Serial.print("AccZ");Serial.print(",");
  Serial.print("GyrX");Serial.print(",");
  Serial.print("GyrY");Serial.print(",");
  Serial.println("GyrZ");
}

void loop() {
  // Dealing with interrupt
  InletPulses = 0;                                                  // Resets pulse counter for inlet flow.
  OutletPulses = 0;                                                 // Resets pulse counter for outlet flow.
  interrupts();                                                     // Enables interrupts.
  delay(interval);
  //noInterrupts();
  // Reading sensors' values.
  valuePiezo1 = analogRead(pinPiezo1);                              // Piezo 1 value.
  valuePiezo2 = analogRead(pinPiezo2);                              // Piezo 2 value.
  value801s = analogRead(pin801s);                                  // 801s value.      
  valueCond = analogRead(pinCond);                                  // Condutance value.
  valuePressure = readDeltaPressure();                              // Pressure value.
  valueInletFlow = Flow(HzPerLiterInlet,InletPulses);      // Inlet flow value.
  valueOutletFlow = Flow(HzPerLiterOutlet,OutletPulses);   // Outlet flow value.
  // Reading MPU 6050 measures
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 14, true); // Solicita os dados ao sensor
  AccX = Wire.read() << 8 | Wire.read(); //0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
  AccY = Wire.read() << 8 | Wire.read(); //0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  AccZ = Wire.read() << 8 | Wire.read(); //0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  Temp = Wire.read() << 8 | Wire.read(); //0x41 (TEMP_OUT_H) & 0x42 (TEMP_OUT_L)
  GyrX = Wire.read() << 8 | Wire.read(); //0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  GyrY = Wire.read() << 8 | Wire.read(); //0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  GyrZ = Wire.read() << 8 | Wire.read(); //0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)
  // Scaling factor MPU 6050
  /* Accel
      +/-2g = 16384
      +/-4g = 8192
      +/-8g = 4096
      +/-16g = 2048

      Gyro
      +/-250°/s = 131
      +/-500°/s = 65.6
      +/-1000°/s = 32.8
      +/-2000°/s = 16.4
  */
  // Accel
  AccX = AccX/16384;
  AccY = AccY/16384;
  AccZ = AccZ/16384;
  // Gyro
  GyrX = GyrX/131;
  GyrY = GyrY/131;
  GyrZ = GyrZ/131;
  // Printing values
  Serial.print(valuePiezo1);Serial.print(",");
  Serial.print(valuePiezo2);Serial.print(",");
  Serial.print(value801s);Serial.print(",");
  Serial.print(valueCond);Serial.print(",");
  Serial.print(valuePressure);Serial.print(",");
  Serial.print(valueInletFlow);Serial.print(",");
  Serial.print(valueOutletFlow);Serial.print(",");
  Serial.print(AccX);Serial.print(",");
  Serial.print(AccY);Serial.print(",");
  Serial.print(AccZ);Serial.print(",");
  Serial.print(GyrX);Serial.print(",");
  Serial.print(GyrY);Serial.print(",");
  Serial.println(GyrZ);
}

void CountInletPulses(){
  InletPulses++;
}

void CountOutletPulses(){
  OutletPulses++;
}

float readDeltaPressure(){
  float sinal = analogRead(pinPressure);
  float dP;
  if(sinal<102){
    dP = -2.0;
    }else{
      if(sinal>921){
        dP = 2.0;
        } else{
          dP = (sinal/1023-0.5)/0.2;
          }
      }
  return dP;
}

double Flow(int CalibrationFactor,volatile double pulse_counts){
  double Measure;
  Measure = pulse_counts/CalibrationFactor/interval*1000;
  return Measure;
}

void setMPU(){
  Wire.begin();
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  // Configuring Gyro Range
  /*
    Wire.write(0b00000000); //  +/-250°/s
    Wire.write(0b00001000); //  +/-500°/s
    Wire.write(0b00010000); //  +/-1000°/s
    Wire.write(0b00011000); //  +/-2000°/s
  */
  Wire.beginTransmission(MPU);
  Wire.write(0x1B); // Gyro register
  Wire.write(0x00000000);  // Set Gyro range
  Wire.endTransmission();

  // Configuring Accel Range
  /*
      Wire.write(0b00000000); //  +/-2g
      Wire.write(0b00001000); //  +/-4g
      Wire.write(0b00010000); //  +/-8g
      Wire.write(0b00011000); //  +/-16g
  */
  Wire.beginTransmission(MPU);
  Wire.write(0x1C); // Accel register
  Wire.write(0b00000000);  // Set Accel range
  Wire.endTransmission();
}
