import RPi.GPIO as GPIO
import time

# Constants
SERVO_PIN = 13  # GPIO18 (Pin 12)
FREQUENCY = 50  # 50Hz frequency (20ms period)
MIN_DUTY_CYCLE = 2.5
MAX_DUTY_CYCLE = 12.5


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, FREQUENCY)
    pwm.start(0)
    return pwm


def set_angle(pwm, angle):
    duty_cycle = MIN_DUTY_CYCLE + \
        (angle / 180.0) * (MAX_DUTY_CYCLE - MIN_DUTY_CYCLE)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(1)


def cleanup(pwm):
    pwm.stop()
    GPIO.cleanup()


if __name__ == "__main__":
    pwm = setup()
    try:
        while True:
            angle = float(input("Enter angle (0-180): "))
            set_angle(pwm, angle)
    except KeyboardInterrupt:
        cleanup(pwm)
