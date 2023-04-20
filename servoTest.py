import RPi.GPIO as GPIO
import time

servo_pin = 13  # Use pin 13 for the servo motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # Set the PWM frequency to 50 Hz
pwm.start(0)


def set_angle(angle):
    duty_cycle = angle / 18 + 2  # Convert angle to duty cycle
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(1)


set_angle(0)   # Set the servo to the initial position (0 degrees)
time.sleep(2)  # Wait for 2 seconds
set_angle(90)  # Set the servo to 90 degrees
time.sleep(2)  # Wait for 2 seconds
set_angle(180)  # Set the servo to 180 degrees

pwm.stop()
GPIO.cleanup()
