import time
import RPi.GPIO as GPIO
import requests
from time import sleep

headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}

# Add a dictionary to store the last position
last_position = {
    'x': 0,
    'y': 0,
    'z': 10,
}


# Constants
SERVO_PIN = 13  # GPIO18 (Pin 12)
FREQUENCY = 50  # 50Hz frequency (20ms period)
MIN_DUTY_CYCLE = 2.5
MAX_DUTY_CYCLE = 12.5
pwm = None


def setup():
    global pwm
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


def close():
    global pwm
    set_angle(pwm, 80)


def open():
    global pwm
    set_angle(pwm, 60)


def move(x=None, y=None, z=None, calibrate=False, home=False, speed=3000):
    global last_position
    if calibrate:
        json_data = {
            'command': f'G28'
        }
    elif home:
        json_data = {
            'command': f'G0 X{0} Y{220} Z{70} F1{3000}'
        }
        # Update the last sent position
        last_position['x'] = 0
        last_position['y'] = 240
        last_position['z'] = 70
    else:
        # Use the last sent position for unspecified axes
        x = last_position['x'] if x is None else x
        y = last_position['y'] if y is None else y
        z = last_position['z'] if z is None else z

        json_data = {
            'command': f'G0 X{x} Y{y} Z{z} F1{speed}'
        }

        # Update the last sent position
        last_position['x'] = x
        last_position['y'] = y
        last_position['z'] = z

    print(requests.post(
        'http://0.0.0.0/api/printer/command', headers=headers, json=json_data))


if __name__ == "__main__":
    pwm = setup()

# move(calibrate=True)
# sleep(10)
move(home=True)
