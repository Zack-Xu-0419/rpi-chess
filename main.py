# Import Libraries
import requests
from time import sleep

# Constants
IP = '10.1.25.244'
cc = [0, 0, 0]

# Movement Functions
headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}


def move(x=0, y=0, z=10, home=False, speed=3000):
    json_data = {
        'command': f'G0 X{x} Y{y} Z{z} F1{speed}',
    }
    if home:
        json_data = {
            'command': f'G28'
        }
    print(requests.post(
        f'http://{IP}/api/printer/command', headers=headers, json=json_data))


# def border_det():
#     input = picamera.

# Calibrate - Remove ALL CHESS PIECES before calling this


def calibrate():
    # Calibrate XYZ Coordinates on 3D printer
    move(home=True)
    # Raise the Z coordinate.
    move()
    # Move the x axis to the center in order to be able to see the red dots around the plate.
    nodet = True
    while nodet:
        # While not detecting the red dots, move the y coordinate up until it sees 4 red dots.
        cc[1] += 1
        move(cc)


# Main Program
move(home=True)
