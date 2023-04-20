import RPi.GPIO as GPIO
import time

servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # Set the PWM frequency to 50 Hz
pwm.start(0)


def set_angle(angle):
    duty_cycle = angle / 18 + 2  # Convert angle to duty cycle
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)


try:
    start_time = time.time()
    while (time.time() - start_time) < 5:  # Loop for 5 seconds
        for angle in range(0, 11, 1):  # Move from 0 to 10 degrees in 1-degree increments
            set_angle(angle)
        for angle in range(10, -1, -1):  # Move from 10 to 0 degrees in 1-degree increments
            set_angle(angle)
finally:
    pwm.stop()
    GPIO.cleanup()
