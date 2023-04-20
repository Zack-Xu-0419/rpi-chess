import RPi.GPIO as GPIO
import time

servo_pin = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # Set PWM frequency to 50Hz
pwm.start(0)


def set_servo_angle(angle):
    pwm.ChangeDutyCycle(angle)
