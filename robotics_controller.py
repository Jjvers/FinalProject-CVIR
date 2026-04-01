"""
Robotics Controller for Smart Door Lock System
================================================
Course: Computer Vision & Robotics Intelligence

Hardware Components (for real deployment):
  - Servo Motor / Solenoid Lock: Door lock mechanism
  - MQ-2 Smoke/Gas Sensor: Fire/smoke detection
  - Buzzer: Audible alarm for fire alert & access denied
  - Red LED: Flashing alert indicator
  - Raspberry Pi / Arduino: Microcontroller

Pin Configuration (example for Raspberry Pi):
  - SERVO_PIN = GPIO 18 (PWM for door servo)
  - BUZZER_PIN = GPIO 23
  - RED_LED_PIN = GPIO 24
  - MQ2_ANALOG_PIN = A0 (via MCP3008 ADC, since RPi has no analog pins)
  - MQ2_DIGITAL_PIN = GPIO 25 (digital threshold output from MQ-2)

MQ-2 Sensor:
  - Detects: Smoke, LPG, Methane, Alcohol, Hydrogen
  - Output: Analog (voltage proportional to gas concentration)
           Digital (HIGH when gas > threshold, adjustable via potentiometer)
  - Typical threshold: 300-500 on analog scale (0-1023)

This module is a SIMULATION layer. In production, replace print()
calls with actual GPIO commands (RPi.GPIO or gpiozero library).
"""

import time
import serial
import threading
from datetime import datetime


class DoorController:
    """
    Simulates a smart door lock controller.
    
    In real hardware, each method would send GPIO signals:
      - open_door()  → Servo rotates to 90° (unlocked position)
      - lock_door()  → Servo rotates to 0° (locked position)
      - buzzer_on()  → GPIO.output(BUZZER_PIN, HIGH)
      - led_blink()  → Thread loop toggling RED_LED_PIN
    """

    # Hardware pin simulation (for documentation)
    SERVO_PIN = 18
    BUZZER_PIN = 23
    RED_LED_PIN = 24
    MQ2_DIGITAL_PIN = 25
    MQ2_THRESHOLD = 400  # Analog threshold for smoke detection

    def __init__(self, port="COM3", baudrate=115200):
        self.door_state = "LOCKED"
        self.fire_alarm_active = False
        self.buzzer_active = False
        self.led_blinking = False
        self.mq2_last_reading = 0
        self.logs = []
        self.port = port
        
        # Try to connect to ESP32 via Serial
        self.serial_conn = None
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2) # Give ESP32 time to reboot after connection
            self._log(f"✅ SERIAL CONNECTED: Hardware link established on {port}")
        except serial.SerialException as e:
            self._log(f"⚠️ SERIAL WARNING: Cannot connect to ESP32 on {port}. Running in Simulation Mode.")

    def _log(self, message):
        """Internal logging with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        print(f"🤖 ROBOTICS: {entry}")

    # ─── Door Control ───────────────────────────────────────────

    def open_door(self, reason="", student_name=""):
        """
        Open the door (unlock).
        
        Real hardware implementation:
          import RPi.GPIO as GPIO
          servo = GPIO.PWM(self.SERVO_PIN, 50)  # 50Hz PWM
          servo.ChangeDutyCycle(7.5)  # 90 degrees = unlocked
        """
        self.door_state = "UNLOCKED"
        msg = f"🔓 DOOR OPENED (Servo → 90°)"
        if student_name:
            msg += f" for {student_name}"
        if reason:
            msg += f" | Reason: {reason}"
            
        # Send Serial command to ESP32
        if self.serial_conn:
            try:
                # Based on user request, sending exactly 'open'
                # Or 'open|Name|Mood' for LCD support
                cmd = f"open\n"
                if student_name:
                    # Guessing mood is Happy for now, as it's passed differently in original code
                    # But doing simple open per user request "open lock aja ganti code nya"
                    cmd = f"open|{student_name}|\n" 
                self.serial_conn.write(cmd.encode('utf-8'))
            except Exception as e:
                self._log(f"Serial Write Error: {e}")

        self._log(msg)
        self._log("⏳ Door will auto-lock in 5 seconds (Servo → 0°)")
        return True

    def lock_door(self):
        """
        Lock the door.
        Real hardware: servo.ChangeDutyCycle(2.5)  # 0 degrees = locked
        """
        self.door_state = "LOCKED"
        
        if self.serial_conn:
            try:
                self.serial_conn.write(b"lock\n")
            except Exception as e:
                self._log(f"Serial Write Error: {e}")
                
        self._log("🔒 DOOR LOCKED (Servo → 0°)")
        return True

    def deny_access(self, reason=""):
        """
        Deny access — keep door locked + short buzzer beep.
        
        Real hardware:
          GPIO.output(BUZZER_PIN, GPIO.HIGH)
          time.sleep(0.3)
          GPIO.output(BUZZER_PIN, GPIO.LOW)
          # Blink red LED 3 times
          for _ in range(3):
              GPIO.output(RED_LED_PIN, GPIO.HIGH)
              time.sleep(0.2)
              GPIO.output(RED_LED_PIN, GPIO.LOW)
              time.sleep(0.2)
        """
        msg = f"🚫 ACCESS DENIED"
        if reason:
            msg += f" | Reason: {reason}"
            
        if self.serial_conn:
            try:
                cmd = f"lock|{reason}\n"
                self.serial_conn.write(cmd.encode('utf-8'))
            except Exception as e:
                self._log(f"Serial Write Error: {e}")
                
        self._log(msg)
        self._log("🔴 Red LED blinking 3x + Buzzer short beep")
        return False

    # ─── MQ-2 Smoke Sensor & Fire Alarm ─────────────────────────

    def read_mq2_sensor(self, analog_value=None):
        """
        Read MQ-2 smoke/gas sensor.
        
        Real hardware (via MCP3008 ADC for Raspberry Pi):
          import spidev
          spi = spidev.SpiDev()
          spi.open(0, 0)
          adc = spi.xfer2([1, (8 + channel) << 4, 0])
          analog_value = ((adc[1] & 3) << 8) + adc[2]
          
        Or using gpiozero:
          from gpiozero import MCP3008
          pot = MCP3008(channel=0)
          analog_value = pot.value * 1023
          
        Digital pin approach (simpler):
          smoke_detected = GPIO.input(MQ2_DIGITAL_PIN)
          # HIGH = smoke above threshold (set by potentiometer on MQ-2 module)
        """
        if analog_value is None:
            analog_value = self.mq2_last_reading
        
        self.mq2_last_reading = analog_value
        
        if analog_value > self.MQ2_THRESHOLD:
            self._log(f"🔥 MQ-2 SMOKE DETECTED! Reading: {analog_value} (threshold: {self.MQ2_THRESHOLD})")
            self.trigger_fire_alarm()
            return True
        
        return False

    def trigger_fire_alarm(self):
        """
        Activate fire/emergency alarm.
        
        Real hardware sequence:
          1. GPIO.output(BUZZER_PIN, GPIO.HIGH)     → Buzzer ON continuously
          2. Start LED blink thread:
               while fire_alarm_active:
                   GPIO.output(RED_LED_PIN, GPIO.HIGH)
                   time.sleep(0.3)
                   GPIO.output(RED_LED_PIN, GPIO.LOW)
                   time.sleep(0.3)
          3. Servo → 90° (unlock all doors)
          4. Send notification to web dashboard via API
        """
        self.fire_alarm_active = True
        self.door_state = "UNLOCKED"
        self.buzzer_active = True
        self.led_blinking = True
        
        self._log("🚨🔥 FIRE ALARM ACTIVATED!")
        self._log("  → MQ-2 sensor triggered (smoke/gas detected)")
        self._log("  → Buzzer ON (GPIO 23 = HIGH, continuous alarm)")
        self._log("  → Red LED blinking (GPIO 24, 0.3s interval)")
        self._log("  → Servo → 90° (ALL DOORS UNLOCKED for evacuation)")
        self._log("  → Web dashboard notification sent")
        return True

    def deactivate_fire_alarm(self):
        """
        Deactivate fire alarm and return to normal mode.
        
        Real hardware:
          GPIO.output(BUZZER_PIN, GPIO.LOW)   → Buzzer OFF
          GPIO.output(RED_LED_PIN, GPIO.LOW)   → LED OFF
          led_blink_thread.stop()
          servo → 0° (re-lock door)
        """
        self.fire_alarm_active = False
        self.buzzer_active = False
        self.led_blinking = False
        
        self._log("✅ Fire alarm deactivated")
        self._log("  → Buzzer OFF (GPIO 23 = LOW)")
        self._log("  → Red LED OFF (GPIO 24 = LOW)")
        self.lock_door()
        return True

    # ─── Status ─────────────────────────────────────────────────

    def get_status(self):
        """Get the current status of the door system."""
        return {
            "door_state": self.door_state,
            "fire_alarm_active": self.fire_alarm_active,
            "buzzer_active": self.buzzer_active,
            "led_blinking": self.led_blinking,
            "mq2_last_reading": self.mq2_last_reading,
            "mq2_threshold": self.MQ2_THRESHOLD,
            "last_logs": self.logs[-10:] if self.logs else [],
        }


# Global controller instance
door_controller = DoorController()
