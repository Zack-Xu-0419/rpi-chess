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


def a_to_b(start_position, end_position):
    # Move the actuator to the start position and close the claw
    actuator_start = goto(start_position)
    move(z=40)
    open()
    move(z=5)
    sleep(20)
    close()
    move(z=40)

    # Move the actuator to the end position
    actuator_end = goto(end_position)

    # Move the actuator down to z=5 and open the claw to drop the piece
    move(z=5)
    sleep(18)
    open()
    move(home=True)


def goto(chess_coordinate, board_bottom_left=(5, 30), board_top_right=(200, 223)):
    x, y = ord(chess_coordinate[0].lower()) - \
        ord('a') + 1, int(chess_coordinate[1])

    x_range = board_top_right[0] - board_bottom_left[0]
    y_range = board_top_right[1] - board_bottom_left[1]

    actuator_x = board_bottom_left[0] + (x - 1) * x_range / 7
    actuator_y = board_bottom_left[1] + (y - 1) * y_range / 7

    move(actuator_x, actuator_y)
    return (actuator_x, actuator_y)


def move(x=None, y=None, z=None, calibrate=False, home=False, speed=3000):
    global last_position
    if calibrate:
        json_data = {
            'command': f'G28'
        }
    elif home:
        json_data = {
            'command': f'G0 X{0} Y{220} Z{40} F1{3000}'
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
# move(home=True)
