import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 2
GPIO.setup(BUTTON_PIN, GPIO.IN)

def button_callback(channel):
    print("Button pressed!")


GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING,
                      callback=button_callback, bouncetime=300)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
