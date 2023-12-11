#include <SoftwareSerial.h>

SoftwareSerial bluetooth(2, 3); // RX, TX
int leftMotor1 = 13;
int leftMotor2 = 12;
int rightMotor1 = 11;
int rightMotor2 = 10;
int orientation = 0; // 0:up, 1:right, 2:down, 3:left


void setup() {
  Serial.begin(9600);
  bluetooth.begin(9600);
  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
}

void moveForward() {
  digitalWrite(rightMotor1, HIGH);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, LOW);

}

void turnRight() {
  digitalWrite(rightMotor1, LOW);
  digitalWrite(rightMotor2, HIGH);
  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, LOW);
  delay(978);
  stopMotors();
}

void turnLeft() {
  digitalWrite(rightMotor1, HIGH);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, LOW);
  digitalWrite(leftMotor2, HIGH);
  delay(978);
  stopMotors();
}

void stopMotors() {
  digitalWrite(rightMotor1, LOW);
  digitalWrite(rightMotor2, LOW);
  digitalWrite(leftMotor1, LOW);
  digitalWrite(leftMotor2, LOW);

}

void process_data(String data) {
  Serial.println("Received data string: " + data);
  /////////////////////////////////////////////////
  // Extract array size
  int delimiterIndex = data.indexOf(',');
  if (delimiterIndex == -1) {
    Serial.println("Invalid data format");
    return;
  }

  int arraySize = data.substring(0, delimiterIndex).toInt();
  Serial.print("Array size: ");
  Serial.println(arraySize);

  // Extract array elements
  int values[arraySize];
  int i = 0;

  char *token = strtok(const_cast<char*>(data.c_str() + delimiterIndex + 1), ",");
  while (token != NULL && i < arraySize) {
    values[i++] = atoi(token);
    token = strtok(NULL, ",");
  }

  // Now 'values' array contains the individual elements
  for (int j = 0; j < i; ++j) {
    ////////////////////////////////// how to move
      if (values[j] == "up") {
        if(orientation == 0){
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 1){
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 2){
          turnRight();
          turnRight();
          moveForward();
          delay(2000);
          stopMotors();
        }else{
          turnRight();
          moveForward();
          delay(2000);
          stopMotors();
        }
        Serial.println("Moving Up");
        // Add code for moving up
    } else if (values[j] == "down") {
      if(orientation == 0){
          turnLeft();
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 1){
          turnRight();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 2){
          moveForward();
          delay(2000);
          stopMotors();
        }else{
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }
        Serial.println("Moving Down");
        // Add code for moving down
    } else if (values[j] == "right") {
      if(orientation == 0){
          turnRight();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 1){
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 2){
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }else{
          turnLeft();
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }
        Serial.println("Moving Right");
        // Add code for moving right
    } else if (values[j] == "left") {
      if(orientation == 0){
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 1){
          turnLeft();
          turnLeft();
          moveForward();
          delay(2000);
          stopMotors();
        }else if(orientation == 2){
          turnRight();
          moveForward();
          delay(2000);
          stopMotors();
        }else{
          moveForward();
          delay(2000);
          stopMotors();
        }
        Serial.println("Moving Left");
        // Add code for moving left
    } else {
       Serial.println("Unknown values[j]");
    }
  }
  
  
}

void loop() {
  if (bluetooth.available() > 0) {
    String data = bluetooth.readStringUntil('\n');
    process_data(data);
  }
}