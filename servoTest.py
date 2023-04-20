import RPi.GPIO as GPIO
import time

servo_pin = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # Set PWM frequency to 50Hz
pwm.start(0)


def set_servo_angle(angle):
    duty_cycle = (angle / 18.0) + 2.5
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)


try:
    while True:
        set_servo_angle(0)  # Move servo to 0 degrees
        time.sleep(5)
        set_servo_angle(90)  # Move servo to 90 degrees
        time.sleep(5)
        set_servo_angle(180)  # Move servo to 180 degrees
finally:
    pwm.stop()
    GPIO.cleanup()
