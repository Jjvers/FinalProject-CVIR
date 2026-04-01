#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

//  WIFI 
const char* ssid = "apaya";
const char* password = "gataulupa";

// OBJECT 
WebServer server(80);
LiquidCrystal_I2C lcd(0x27, 16, 2);

// PIN 
#define RELAY_PIN 26
#define LED_RED 27
#define LED_GREEN 14
#define BUZZER_PIN 25
#define MQ2_PIN 34

// VARIABLE 
int thresholdGas = 2000;
bool isEmergency = false;
bool isDoorOpen = false;



// BUZZER
void beepBuzzer(int times) {
  for (int i = 0; i < times; i++) {
    tone(BUZZER_PIN, 2000);
    delay(150);
    noTone(BUZZER_PIN);
    delay(100);
  }
}

// OPEN
// Overloaded to accept student Name and Mood to display on LCD
void openDoor(String studentName = "Web Dashboard", String mood = "", String studentId = "") {
  if (isEmergency) return;

  isDoorOpen = true;

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Assuming HIGH unlocks the solenoid

  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_GREEN, HIGH);

  // FRAME 1: Access Granted & ID
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Access Granted!");
  
  lcd.setCursor(0, 1);
  if (studentId != "") {
      lcd.print("ID: " + studentId);
  } else {
      lcd.print("Welcome!");
  }

  beepBuzzer(2);
  delay(1500); // Hold frame 1 for 1.5 seconds

  // FRAME 2: Name & Mood
  lcd.clear();
  lcd.setCursor(0, 0);
  if (studentName.length() > 16) studentName = studentName.substring(0, 16);
  lcd.print(studentName);

  lcd.setCursor(0, 1);
  if (mood != "") {
      String mStr = "Mood: " + mood;
      if (mStr.length() > 16) mStr = mStr.substring(0, 16);
      lcd.print(mStr);
  }
}

// CLOSE
void closeDoor(String reason = "") {
  if (isEmergency) return;

  isDoorOpen = false;

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // Lock solenoid

  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, LOW);

  lcd.clear();
  lcd.setCursor(0, 0);
  if (reason != "") {
    lcd.print("ACCESS DENIED");
    if(reason.length() > 16) reason = reason.substring(0, 16);
    lcd.setCursor(0, 1);
    lcd.print(reason);
    beepBuzzer(3); // Error beeps
  } else {
    lcd.print("Door Locked");
    lcd.setCursor(0, 1);
    lcd.print("Waiting 4 Scan");
    beepBuzzer(1); // Normal lock beep
  }
}

// EMERGENCY
void triggerEmergency() {
  if (isEmergency) return; // Prevent continuous overriding
  
  isEmergency = true;
  isDoorOpen = true;

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Unlock Door for safety
  
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, LOW);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("EMERGENCY!");
  lcd.setCursor(0, 1);
  lcd.print("GAS/FIRE DETECTED");

  Serial.println("!!! EMERGENCY !!!");
  
  // Note: This blocks main code, meaning emergency stops all camera operations
  // To reset, device must be manually rebooted.
  while (true) {
    server.handleClient(); // Still allow checking Web Dashboard status
    for (int hz = 500; hz < 2500; hz += 20) {
      tone(BUZZER_PIN, hz);
      delay(2);
    }
    for (int hz = 2500; hz > 500; hz -= 20) {
      tone(BUZZER_PIN, hz);
      delay(2);
    }
  }
}


// SETUP
void setup() {
  Serial.begin(115200);

  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(MQ2_PIN, INPUT);
  
  // Initial lock state
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, LOW);

  // LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");

  // WIFI
  WiFi.begin(ssid, password);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("IP:");
  lcd.setCursor(0, 1);
  lcd.print(WiFi.localIP());
  delay(3000);

  lcd.clear();
  lcd.print("Waiting 4 Scan");



  server.on("/open", []() {
    openDoor();
    server.send(200, "text/plain", "OPEN");
  });

  server.on("/close", []() {
    closeDoor();
    server.send(200, "text/plain", "LOCKED");
  });

  server.on("/status", []() {
    if (isEmergency) server.send(200, "text/plain", "EMERGENCY!");
    else server.send(200, "text/plain", isDoorOpen ? "OPEN" : "LOCKED");
  });

  server.begin();
  Serial.println("Dashboard Server Started!");
}

// LOOP
void loop() {
  server.handleClient();

  // Python Laptop Serial Control
  // Listens via USB cable from your laptop running the face tracking
  if (Serial.available() && !isEmergency) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    // Command formats expected from python:
    // "open|Johana|Happy|10113" 
    // "lock|No face detected"
    // "lock"
    
    int firstDivider = cmd.indexOf('|');
    if (firstDivider != -1) {
       String action = cmd.substring(0, firstDivider);
       String remainder = cmd.substring(firstDivider + 1);
       
       if (action == "open") {
          int secondDivider = remainder.indexOf('|');
          if (secondDivider != -1) {
              String sName = remainder.substring(0, secondDivider);
              String rem2 = remainder.substring(secondDivider + 1);
              
              int thirdDivider = rem2.indexOf('|');
              if (thirdDivider != -1) {
                  String sMood = rem2.substring(0, thirdDivider);
                  String sId = rem2.substring(thirdDivider + 1);
                  openDoor(sName, sMood, sId); 
              } else {
                  openDoor(sName, rem2); 
              }
          } else {
              openDoor(remainder); 
          }
       } else if (action == "lock") {
          closeDoor(remainder); // remainder = Error reason
       }
    } else {
       // Backup flat commands like original
       if (cmd == "open") openDoor();
       if (cmd == "lock") closeDoor();
    }
  }

  // MQ2 Gas Sensor Control
  if (!isEmergency) {
    int gas = analogRead(MQ2_PIN);
    if (gas > thresholdGas) {
      triggerEmergency();
    }
  }
}
