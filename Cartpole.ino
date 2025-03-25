#include <Wire.h>
#include <Motoron.h>

// ====== ENCODER & PID SETTINGS ======
#define ENCODER_A 2
#define ENCODER_B 3
#define CPR       2000    // Counts Per Revolution for your encoder

// PID gains
float Kp = 209.0;
float Ki = 0.0;
float Kd = 0.0;

// Variables for PID
volatile long encoder_value = 0; 
float previous_error = 0;
float integral = 0;

// Desired angle
float setpoint = 0.0;

// 80% of full Motoron range (±3200)
const int MAX_SPEED = 2560;

// ====== TWO MOTORON SHIELDS ======
// 0x10 for Front Shield, 0x11 for Rear Shield
MotoronI2C motoronFront(0x10);
MotoronI2C motoronRear(0x11);

// ====== INTERRUPT SERVICE ROUTINE ======
void encoder_isr() {
  int A = digitalRead(ENCODER_A);
  int B = digitalRead(ENCODER_B);

  // Quadrature decode
  if ((A == HIGH) != (B == LOW)) {
    encoder_value--;
  } else {
    encoder_value++;
  }
}

void setup() {
  Serial.begin(9600);
  Wire.begin();

  // ====== ENCODER PINS ======
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoder_isr, CHANGE);

  // ====== INIT FRONT SHIELD (0x10) ======
  Serial.println("Initializing Motoron Front (0x10)...");
  motoronFront.reinitialize();
  motoronFront.disableCrc();
  motoronFront.clearResetFlag();

  uint8_t frontFlags = motoronFront.getResetFlag();
  Serial.print("Front Reset Flags: ");
  Serial.println(frontFlags, BIN);

  // Set acceleration/deceleration for channels 1 and 2
  motoronFront.setMaxAcceleration(1, 1024);
  motoronFront.setMaxDeceleration(1, 1024);
  motoronFront.setMaxAcceleration(2, 1024);
  motoronFront.setMaxDeceleration(2, 1024);

  // ====== INIT REAR SHIELD (0x11) ======
  Serial.println("Initializing Motoron Rear (0x11)...");
  motoronRear.reinitialize();
  motoronRear.disableCrc();
  motoronRear.clearResetFlag();

  uint8_t rearFlags = motoronRear.getResetFlag();
  Serial.print("Rear Reset Flags: ");
  Serial.println(rearFlags, BIN);

  motoronRear.setMaxAcceleration(1, 1024);
  motoronRear.setMaxDeceleration(1, 1024);
  motoronRear.setMaxAcceleration(2, 1024);
  motoronRear.setMaxDeceleration(2, 1024);

  delay(1000);
  Serial.println("Setup complete!");
}

void loop() {
  // ====== 1) Compute Current Angle ======
  float angle = (float(encoder_value) / CPR) * 360.0;

  // Keep angle in [-180, 180]
  if (angle > 180.0)  angle -= 360.0;
  if (angle < -180.0) angle += 360.0;

  // ====== 2) PID Control ======
  float error = setpoint - angle;
  integral += error;
  float derivative = error - previous_error;
  float output = Kp * error + Ki * integral + Kd * derivative;
  if (error>35 | error<-35){
    output = 0;}
  previous_error = error;

  // ====== 3) Scale Output for 80% Speed ======
  // Motoron range is ±3200, so 80% is ±2560
  int motorSpeed = (int)constrain(output, -MAX_SPEED, MAX_SPEED);

  // ====== 4) Differential Drive ======
  //  - Left motors = +motorSpeed
  //  - Right motors = -motorSpeed
  // Front Shield (M1=Front Left, M2=Front Right)
  motoronFront.setSpeed(1,  motorSpeed);   // Front Left
  motoronFront.setSpeed(2, -motorSpeed);   // Front Right

  // Rear Shield (M1=Rear Left, M2=Rear Right)
  motoronRear.setSpeed(1,   -motorSpeed);   // Rear Left
  motoronRear.setSpeed(2,  motorSpeed);   // Rear Right

  // ====== 5) Debug Info ======
  //Serial.print("Angle: ");
  //Serial.print(angle, 2);
  //Serial.print("\n");
  //Serial.print("  |  PID Output: ");
  //Serial.print(output, 2);
  //Serial.print("  |  Motor Speed: ");
  //Serial.println(motorSpeed);

  // ====== 6) Short Delay ======
  delay(1);  // ~100 Hz loop
}