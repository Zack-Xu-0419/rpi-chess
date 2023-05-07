# Import Libraries
import requests
import cv2 as cv
import numpy as np
from time import sleep
import picamera
from picamera.array import PiRGBArray
import pprint
import chess
import stockfish
from collections import OrderedDict
import time
import RPi.GPIO as GPIO
import requests
from time import sleep
import threading

headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}

# Add a dictionary to store the last position
last_position = {
    'x': 0,
    'y': 220,
    'z': 40,
}


# Constants
SERVO_PIN = 13
FREQUENCY = 50  # 50Hz frequency (20ms period)
MIN_DUTY_CYCLE = 2.5
MAX_DUTY_CYCLE = 12.5
pwm = None

OFFSET_X = 0
OFFSET_Y = +5

# Constants
IP = '0.0.0.0'
cc = [120, 120, 0]
previousRes = []

previousBoard = chess.Board()

board = chess.Board()
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
fish = stockfish.Stockfish("../../Stockfish-sf_15/src/stockfish")

# Set skill level (0 to 20)
fish.set_skill_level(5)

# Set contempt value (positive for aggressive play, negative for defensive play)
fish.update_engine_parameters({'Contempt': 20})


# Movement Functions
headers = {
    'Content-type': 'application/json',
    'X-Api-Key': '041AA6FA66184165A38B8D938C68A30E',
}


def edge_det(output):
    final_res = []
    # with picamera.PiCamera() as camera:
    #     camera.resolution = (640, 480)
    #     rawCapture = PiRGBArray(camera)
    #     camera.capture(rawCapture, format="bgr")
    #     output = rawCapture.array
    #     camera.close()

    hsv_img = cv.cvtColor(output, cv.COLOR_RGB2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([100, 100, 100])
    upper2 = np.array([120, 255, 255])

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
        # If is toward the center, throw it away
        centerLeftT = 640/2-50
        centerRightT = 640/2+50
        if i[0] < centerLeftT or i[0] > centerRightT:
            final_res.append(i)
    # if len(final_res) != 4:
    #     return "ERROR"

    # Find bottom left and top right (The position that have: (lowY, highX), (highY, lowX))
    ffinal_res = [0, 0]
    for i in final_res:
        if(i[0] > 400 and i[1] < 200):
            ffinal_res[1] = i
        if(i[1] > 400 and i[0] < 200):
            ffinal_res[0] = i

    return ffinal_res


def crop(input):
    # Read input image
    img = input

    # Define 4 points for the crop using the new coordinates
    # Top left
    x1, y1 = 240, 7
    # Top right
    x2, y2 = 1031, 7
    # Bottom right
    x3, y3 = 1033, 792
    # Bottom left
    x4, y4 = 216, 786

    src_pts = np.array(
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # Calculate the dimensions of the output image
    width = max(int(np.sqrt((x2-x1)**2 + (y2-y1)**2)),
                int(np.sqrt((x3-x4)**2 + (y3-y4)**2)))
    height = max(int(np.sqrt((x4-x1)**2 + (y4-y1)**2)),
                 int(np.sqrt((x3-x2)**2 + (y3-y2)**2)))

    # Define destination points for the transformation
    dst_pts = np.array(
        [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

    # Get the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation to the input image
    transformed_img = cv.warpPerspective(img, M, (width, height))

    return transformed_img


def captureImage():
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 960)
        rawCapture = PiRGBArray(camera)
        camera.capture(rawCapture, format="bgr")
        output = rawCapture.array
        camera.close()
    cv.imwrite("./originalOut.jpg", output)


def getBoardState(output, edges=[0, 0, 0, 0]):

    output = crop(output)

    # if edges[0] != 0:
    #     output = output[edges[0]:edges[1], edges[2]:edges[3]]
    # else:
    #     detectedEdges = edge_det(output=output)
    #     output = output[int(detectedEdges[1][1]):int(detectedEdges[0][1]), int(
    #         detectedEdges[0][0]):int(detectedEdges[1][0])]

    hsv_img = cv.cvtColor(output, cv.COLOR_BGR2HSV)

    cv.imwrite("out.jpg", hsv_img)

    # Define the range of green color in HSV
    lower_green = np.array([50, 58, 58])
    upper_green = np.array([200, 255, 255])

    # Create a mask for green color
    green_mask = cv.inRange(hsv_img, lower_green, upper_green)

    # Apply the mask to the original image
    result = cv.bitwise_and(output, output, mask=green_mask)

    gray = cv.cvtColor(cv.cvtColor(
        result, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

    gray_orig = cv.cvtColor(cv.cvtColor(
        hsv_img, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

    # Save output:
    origCropped = gray

    edges = cv.Canny(gray, 50, 100)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1,
                              minRadius=10, maxRadius=30, param2=18, minDist=30)

    # print(len(circles))
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # print(x, y, r)
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(output, (x, y), r, (0, 255, 0), 4)
            cv.rectangle(output, (x - 5, y - 5),
                         (x + 5, y + 5), (0, 128, 255), -1)

    # Drawing the pixles to divide up the grid, development purposes only
    dify = int(len(output)/8)
    difx = int(len(output[0])/8)+3
    const = 0
    consty = +4
    for i in range(8):
        for j in range(8):
            cv.rectangle(output, (const+i * difx, j * dify+consty),
                         (const+(i+1) * difx, (j+1) * dify+consty), (0, 0, 255), 1)
    cv.imwrite("../output.jpg", output)

    # Start from the top.

    board = []
    bigDiff = OrderedDict()
    for j in range(8):
        # For each row
        row = []
        for i in range(8):
            # Compare the Color of each square of previous image vs current:
            # print(len(previousImg))

            rect = [const+i * difx, j * dify+consty, const+(i+1) * difx,
                    (j+1) * dify+consty]  # xMin, yMin, xMax, yMax
            # check if a center of a circle is in that range
            isSomething = False
            for (x, y, r) in circles:
                # If there is no piece
                if x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3]:
                    isSomething = True
                    row.append(0)
            if not isSomething:
                # If there is something, detect if it is white or black
                curr = gray_orig[int(j * dify + consty + dify/2-10): int((j+1) * dify+consty - dify/2+10), int(const+i *
                                 difx + difx/2-10):int(const+(i+1) * difx-difx/2+10)]
                avg = np.mean(curr)
                # print(avg)
                if avg < 110:
                    row.append(2)
                else:
                    row.append(1)
        board.append(row)

    return board


def getBoardDiff(input):
    a = []
    b = []
    is_castling = False

    for i in range(len(input)):
        for j in range(len(input[0])):
            if previousRes[i][j] == 1 and input[i][j] == 0:
                a.append(i)
                a.append(j)
            if previousRes[i][j] == 2 and input[i][j] == 1:
                b.append(i)
                b.append(j)
            if previousRes[i][j] == 0 and input[i][j] == 1:
                b.append(i)
                b.append(j)

    a_groups = [a[i:i+2] for i in range(0, len(a), 2)]
    b_groups = [b[i:i+2] for i in range(0, len(b), 2)]

    if len(a_groups) == 2 and len(b_groups) == 2:
        king_from = []
        king_to = []

        if [7, 4] in a_groups:
            king_from = [7, 4]
            if [7, 6] in b_groups:
                king_to = [7, 6]
                is_castling = True
                return ("e1g1", is_castling)  # white kingside castling
            elif [7, 2] in b_groups:
                king_to = [7, 2]
                is_castling = True
                return ("e1c1", is_castling)  # white queenside castling

    return ((a, b), is_castling)


def rundet():
    global previousRes
    global previousBoard
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 960)
        rawCapture = PiRGBArray(camera)
        camera.capture(rawCapture, format="bgr")
        output = rawCapture.array
        camera.close()
    fromBlack = getBoardState(output)

    fromWhite = []
    pieceFrom = [-1, -1]
    for i in fromBlack[::-1]:
        fromWhite.append(i[::-1])
    pprint.pprint(fromWhite)
    if previousRes != []:
        difference, is_castling = getBoardDiff(fromWhite)
        # print(is_castling)
        if not is_castling:
            pieceFrom = [letters[(difference[0][1])], 8-difference[0][0]]
            pieceTo = [letters[(difference[1][1])], 8-difference[1][0]]

            finalCommand = ""

            # print("PIECEFROM:")
            # print(pieceFrom)
            # print("PIECETO?:")
            # print(pieceTo)
            finalCommand = f"{pieceFrom[0]}{pieceFrom[1]}{pieceTo[0]}{pieceTo[1]}"
        else:
            finalCommand = difference  # castling move

        # print(finalCommand)
        board.push(chess.Move.from_uci(finalCommand))
        previousBoard = board.copy()

    previousRes = fromWhite
    # print(board)


def getMove():
    fish.set_fen_position(board.fen())
    best_move = fish.get_best_move_time(1000)
    fish.set_depth(10)
    print("EVAL:" + f"{fish.get_evaluation()['value'] / 100:+.2f}")
    fish.set_depth(20)
    board.push(chess.Move.from_uci(best_move))
    letter1 = best_move[0]
    number1 = best_move[1]
    letter2 = best_move[2]
    number2 = best_move[3]
    for i in range(len(letters)):
        if letters[i] == (letter1):
            letter1 = i
    for i in range(len(letters)):
        if letters[i] == (letter2):
            letter2 = i

    # print(letter1, number1)
    # previousRes[8-int(number1)][letter1] = 0
    # print(letter2, number2)
    # previousRes[8-int(number2)][letter2] = 2
    # pprint.pprint(previousRes)

    return best_move
    # Track moves:

    # move(home=True)
    # move(z=50)
    # move(boardForward=True)
    # sleep(10)


def DetectAndThink():
    rundet()
    computerMove = getMove()
    # print(board)
    # print("PREVIOUS:")
    # print(previousBoard)
    print(computerMove)
    return computerMove


def process_input(input_str):
    start_position = input_str[:2]
    end_position = input_str[2:]
    return start_position, end_position


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
    set_angle(pwm, 70)


def open():
    global pwm
    set_angle(pwm, 50)


TOP_Z = 50
BOTTOM_Z = 7
SLEEP_BEFORE_CLOSE = 3
SLEEP_AFTER_CLOSE = 3
SLEEP_BEFORE_OPEN = 4
SLEEP_AT_END = 2


def move_piece_off(end_position, addDelay):
    goto(end_position)
    move(z=BOTTOM_Z)
    sleep(SLEEP_BEFORE_CLOSE + addDelay)
    close()
    move(z=TOP_Z)
    sleep(SLEEP_AFTER_CLOSE)
    move(x=240)
    sleep(2)
    open()


def grab_piece(start_position):
    actuator_start = goto(start_position)
    open()
    move(z=BOTTOM_Z)
    sleep(SLEEP_BEFORE_CLOSE)
    close()
    move(z=TOP_Z)


def drop_piece(end_position):
    actuator_end = goto(end_position)
    move(z=BOTTOM_Z)
    sleep(SLEEP_BEFORE_OPEN)
    open()
    move(home=True)


def handle_castling(start_position, end_position):
    if start_position == "e8" and end_position == "c8":
        a_to_b("a8", "d8", addDelay=5)
    elif start_position == "e8" and end_position == "g8":
        a_to_b("h8", "f8", addDelay=5)


def a_to_b(start_position, end_position, addDelay=0):
    time.sleep(addDelay)
    start_square = chess.SQUARE_NAMES.index(start_position)
    end_square = chess.SQUARE_NAMES.index(end_position)

    is_occupied = previousBoard.piece_at(end_square) is not None
    # print(is_occupied)
    # print(previousBoard.piece_at(end_square))

    move(z=TOP_Z)

    if is_occupied:
        move_piece_off(end_position, addDelay)

    grab_piece(start_position)

    drop_piece(end_position)

    handle_castling(start_position, end_position)


def goto(chess_coordinate, board_bottom_left=(5, 30), board_top_right=(200, 223)):
    x, y = ord(chess_coordinate[0].lower()) - \
        ord('a') + 1, int(chess_coordinate[1])

    x_range = board_top_right[0] - board_bottom_left[0]
    y_range = board_top_right[1] - board_bottom_left[1]

    actuator_x = board_bottom_left[0] + (x - 1) * x_range / 7 + OFFSET_X
    actuator_y = board_bottom_left[1] + (y - 1) * y_range / 7 + OFFSET_Y

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
            'command': f'G0 Z{TOP_Z} F1{3000}'
        }

        requests.post(
            'http://0.0.0.0/api/printer/command', headers=headers, json=json_data)

        sleep(0.3)

        json_data = {
            'command': f'G0 X{0} Y{220} Z{TOP_Z} F1{3000}'
        }
        # Update the last sent position
        last_position['x'] = 0
        last_position['y'] = 240
        last_position['z'] = TOP_Z
        requests.post(
            'http://0.0.0.0/api/printer/command', headers=headers, json=json_data)

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


def det_think_move():
    move = DetectAndThink()
    a_to_b(move[:2], move[2:])


def setSpeed():
    json_data = {
        'command': "M203 Z 10000"
    }
    requests.post(
        'http://0.0.0.0/api/printer/command', headers=headers, json=json_data)


GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 2
GPIO.setup(BUTTON_PIN, GPIO.IN)


# def button_callback():
#     print("Button pressed")
#     det_think_move()


# GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING,
#                       callback=button_callback, bouncetime=300)


def button_monitor():
    while True:
        if GPIO.input(BUTTON_PIN) == 0:  # Change to '1' if using a pull-down resistor
            print("Thinking...")
            # Add a debounce delay to avoid multiple detections
            det_think_move()
            time.sleep(0.3)


def quit():
    print("Quitting")
    cleanup(pwm)
    button_thread.join()
    exit(0)


def blitz():
    while True:
        det_think_move()
        for i in range(6):
            sleep(1)
            print(20 - i - 1)


if __name__ == "__main__":
    pwm = setup()
    setSpeed()
    button_thread = threading.Thread(target=button_monitor)
    button_thread.start()

# move(calibrate=True)
# sleep(10)
# move(home=True)
