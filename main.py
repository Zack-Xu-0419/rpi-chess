# Import Libraries
import requests
import cv2 as cv
import numpy as np
from time import sleep
import picamera
from picamera.array import PiRGBArray

# Constants
IP = '0.0.0.0'
cc = [120, 120, 0]

# Movement Functions
headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}


def move(x=0, y=0, z=10, home=False, speed=3000, boardForward=False):
    if x > 240 or x < 0:
        return 0

    if y > 240 or y < 0:
        return 0

    if z > 50 or z < 0:
        return 0

    json_data = {
        'command': f'G0 X{x} Y{y} Z{z} F1{speed}',
    }
    if home:
        json_data = {
            'command': f'G28'
        }
    if boardForward:
        json_data = {
            'command': f'G0 X0 Y240'
        }
    print(requests.post(
        f'http://{IP}/api/printer/command', headers=headers, json=json_data))


def edge_det():
    final_res = []
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        rawCapture = PiRGBArray(camera)
        camera.capture(rawCapture, format="bgr")
        output = rawCapture.array
        camera.close()

    hsv_img = cv.cvtColor(output, cv.COLOR_RGB2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv.inRange(hsv_img, lower1, upper1)
    upper_mask = cv.inRange(hsv_img, lower2, upper2)

    full_mask = lower_mask + upper_mask

    # Apply the mask to the original image
    result = cv.bitwise_and(output, output, mask=full_mask)

    gray = cv.cvtColor(cv.cvtColor(
        result, cv.COLOR_HSV2RGB), cv.COLOR_RGB2GRAY)

    contours, _ = cv.findContours(gray, 1, 2)

    avg = np.zeros((len(contours), 2))

    for i in range(len(contours)):
        a = np.array([0, 0])
        counter = 0
        for j in contours[i]:
            counter += 1
            a[0] += j[0][0]
            a[1] += j[0][1]
        a = a/counter
        avg[i] = (a)

    # print(list(avg))

    for i in list(avg):
        print(i)
        # If is toward the center, throw it away
        centerLeftT = 640/2-50
        centerRightT = 640/2+50
        if i[0] < centerLeftT or i[0] > centerRightT:
            final_res.append(i)
    return final_res


# Calibrate - Remove ALL CHESS PIECES before calling this
# def calibrate():
#     # Calibrate XYZ Coordinates on 3D printer
#     move(home=True)
#     # Raise the Z coordinate.
#     # Move the x axis to the center in order to be able to see the red dots around the plate.
#     move()
#     sleep(5)
#     nodet = True
#     while nodet:
#         # While not detecting the red dots, move the y coordinate up until it sees 4 red dots.
#         cc[1] += 3
#         move(cc[0], cc[1], cc[2])
#         sleep(0.5)
#         if len(edge_det()) == 4:
#             nodet = False


# Main Program
# calibrate()
move(home=True)
move(boardForward=True)
