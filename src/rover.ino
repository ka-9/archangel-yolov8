int leftMotor1 = 13;
int leftMotor2 = 12;
int rightMotor1 = 11;
int rightMotor2 = 10;
int leftMotor3 = 9;
int leftMotor4 = 8;
int rightMotor3 = 7;
int rightMotor4 = 6;

void setup() {
  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
  pinMode(rightMotor3, OUTPUT);
  pinMode(rightMotor4, OUTPUT);
  pinMode(leftMotor3, OUTPUT);
  pinMode(leftMotor4, OUTPUT);
}

void moveForward() {
  digitalWrite(rightMotor1, HIGH);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, LOW);
  digitalWrite(rightMotor3, HIGH);
  digitalWrite(rightMotor4, LOW);
  digitalWrite(leftMotor3, HIGH);
  digitalWrite(leftMotor4, LOW);
}

void moveBackward() {
  digitalWrite(rightMotor1, LOW);
  digitalWrite(rightMotor2, HIGH);
  digitalWrite(leftMotor1, LOW);
  digitalWrite(leftMotor2, HIGH);
  digitalWrite(rightMotor3, LOW);
  digitalWrite(rightMotor4, HIGH);
  digitalWrite(leftMotor3, LOW);
  digitalWrite(leftMotor4, HIGH);
}

void moveRight() {
  digitalWrite(rightMotor1, HIGH);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, LOW);
  digitalWrite(leftMotor2, HIGH);
  digitalWrite(rightMotor3, HIGH);
  digitalWrite(rightMotor4, LOW);
  digitalWrite(leftMotor3, LOW);
  digitalWrite(leftMotor4, HIGH);
}

void moveLeft() {
  digitalWrite(rightMotor1, LOW);
  digitalWrite(rightMotor2, HIGH);
  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, LOW);
  digitalWrite(rightMotor3, LOW);
  digitalWrite(rightMotor4, HIGH);
  digitalWrite(leftMotor3, HIGH);
  digitalWrite(leftMotor4, LOW);
}

void stopMotors() {
  digitalWrite(rightMotor1, LOW);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, LOW);
  digitalWrite(leftMotor2, LOW);
  digitalWrite(rightMotor3, LOW);
  digitalWrite(rightMotor4, LOW);
  digitalWrite(leftMotor3, LOW);
  digitalWrite(leftMotor4, LOW);
}

void loop() {
  // Move forward for 2 seconds
  moveForward();
  delay(2000);

  // Stop for 1 second
  stopMotors();
  delay(1000);

  // Move backward for 2 seconds
  moveBackward();
  delay(2000);

  // Stop for 1 second
  stopMotors();
  delay(1000);

  // Move right for 2 seconds
  moveRight();
  delay(2000);

  // Stop for 1 second
  stopMotors();
  delay(1000);

  // Move left for 2 seconds
  moveLeft();
  delay(2000);

  // Stop for 1 second
  stopMotors();
  delay(1000);
}