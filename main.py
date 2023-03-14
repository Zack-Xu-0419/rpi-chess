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

# Constants
IP = '0.0.0.0'
cc = [120, 120, 0]
previousRes = []
previousImg = np.array([])

board = chess.Board()
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
fish = stockfish.Stockfish("../../Stockfish-sf_15/src/stockfish")


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


def getBoardState(output, edges=[0, 0, 0, 0]):
    global previousImg
    global previousRes

    detectedEdges = edge_det(output=output)
    if edges[0] != 0:
        output = output[edges[0]:edges[1], edges[2]:edges[3]]
    else:
        output = output[int(detectedEdges[1][1]):int(detectedEdges[0][1]), int(
            detectedEdges[0][0]):int(detectedEdges[1][0])]

    hsv_img = cv.cvtColor(output, cv.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([50, 58, 58])
    upper_green = np.array([200, 255, 255])

    # Create a mask for green color
    green_mask = cv.inRange(hsv_img, lower_green, upper_green)

    # Apply the mask to the original image
    result = cv.bitwise_and(output, output, mask=green_mask)

    gray = cv.cvtColor(cv.cvtColor(
        result, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

    # Save output:
    origCropped = gray

    edges = cv.Canny(gray, 50, 100)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1,
                              minRadius=5, maxRadius=25, param2=15, minDist=20)

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
    difx = int(len(output[0])/8)
    const = 10
    for i in range(8):
        for j in range(8):
            cv.rectangle(output, (const+i * difx, j * dify),
                         (const+(i+1) * difx, (j+1) * dify), (0, 0, 255), 1)

    # Start from the top.

    board = []
    bigDiff = []
    for j in range(8):
        # For each row
        row = []
        for i in range(8):
            # Compare the Color of each square of previous image vs current:
            # print(len(previousImg))
            try:
                prev = previousImg[j * dify:(j+1) * dify,
                                   const+i * difx:const+(i+1) * difx]
                curr = gray[j * dify:(j+1) * dify, const+i *
                            difx:const+(i+1) * difx]
                dif = abs(np.mean(curr) - np.mean(prev))
                # print(dif)
                if abs(np.mean(curr) - np.mean(prev)) > 1:
                    print(f"{i}::::{j}")
                    print(dif)
                    bigDiff.append([i, j])
                elif abs(np.mean(curr) - np.mean(prev)) > 0.2 and j == 7 and i == 7:
                    print(f"{i}::::{j}")
                    print(dif)
                    bigDiff.append([i, j])
            except:
                pass

            rect = [const+i * difx, j * dify, const+(i+1) * difx,
                    (j+1) * dify]  # xMin, yMin, xMax, yMax
            # check if a center of a circle is in that range
            isSomething = False
            for (x, y, r) in circles:
                if x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3]:
                    isSomething = True
                    row.append(0)
            if not isSomething:
                row.append(1)
        board.append(row)

    # After comparing, set the previmage to curr image
    previousImg = gray

    return board, bigDiff


def getBoardDiff(input):
    # Looks at which square turned from 1 to 0
    r = []
    # pprint.pprint(previousRes)
    # pprint.pprint(input)
    for i in range(len(input)):
        for j in range(len(input[0])):
            if previousRes[i][j] == 1 and input[i][j] == 0:
                r.append(i)
                r.append(j)
    return r


def rundet():
    global previousRes
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 960)
        rawCapture = PiRGBArray(camera)
        camera.capture(rawCapture, format="bgr")
        output = rawCapture.array
        camera.close()
    detected = getBoardState(output, edges=[45*2, 436*2, 125*2, 539*2])
    fromBlack = detected[0]
    pieceTo = detected[1]
    # convert the two different movement of piece to uci
    pieceTo1 = [letters[-(pieceTo[0][0]-8)-1], pieceTo[0][1]+1]
    print(-(pieceTo[0][0]-8)-1, pieceTo[0][1]+1)
    pieceTaken = False

    finalCommand = ""
    try:
        pieceTo2 = [letters[-(pieceTo[1][0]-8)-1], pieceTo[1][1]+1]
        print(-(pieceTo[1][0]-8)-1, pieceTo[1][1]+1)
    except:
        pieceTaken = True

    fromWhite = []
    pieceFrom = [-1, -1]
    for i in fromBlack[::-1]:
        fromWhite.append(i[::-1])
    if previousRes != []:
        pieceFrom = getBoardDiff(fromWhite)
        # print(pieceFrom.__str__() + "FROM")
        pieceFrom = f"{letters[pieceFrom[1]]}{8-pieceFrom[0]}"

    if pieceTaken:
        finalCommand = f"{pieceFrom}{pieceTo1}"
    else:

        # print(pieceTo1, pieceTo2)
        pieceTo1 = f"{pieceTo1[0]}{pieceTo1[1]}"
        pieceTo2 = f"{pieceTo2[0]}{pieceTo2[1]}"

        if pieceTo1 == pieceFrom:
            finalCommand = f"{pieceFrom}{pieceTo2}"
        elif pieceTo2 == pieceFrom:
            finalCommand = f"{pieceFrom}{pieceTo1}"
    previousRes = fromWhite
    board.push(chess.Move.from_uci(finalCommand))
    print(board)
    return fromWhite, pieceFrom, pieceTo, finalCommand


def getMove():
    fish.set_fen_position(board.fen())
    best_move = fish.get_best_move()
    return best_move
    # Track moves:

    # move(home=True)
    # move(z=50)
    # move(boardForward=True)
    # sleep(10)
