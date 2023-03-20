import cv2 as cv
import numpy as np
import pprint
# import picamera

# # Get Image
# with picamera.PiCamera() as camera:
#     camera.resolution = (640, 480)
#     output = np.empty((640, 480, 3))
#     camera.capture(output, 'rgb')
#     camera.close()

# Returns bottom left of the board and top right of the board


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
    cv.imshow("r", result)

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
    # if len(final_res) != 4:
    #     return "ERROR"

    # Find bottom left and top right (The position that have: (lowY, highX), (highY, lowX))
    ffinal_res = [0, 0]
    for i in final_res:
        if(i[0] > 400 and i[1] < 200):
            ffinal_res[1] = i
        if(i[1] > 400 and i[0] < 200):
            ffinal_res[0] = i
    pprint.pprint(ffinal_res)
    print("***")
    return ffinal_res


def crop(input):
    # Read input image
    img = input

    # Define 4 points for the crop
    # Top left
    distort_x1 = 6
    distort_x2 = 8
    distort_y1 = 0
    distort_y2 = 5

    x1 = 140
    x2 = 540
    y1 = 40
    y2 = 430

    src_pts = np.array(
        [[x1+distort_x1, y1+distort_y1], [x2-distort_x2, y1+distort_y2],
         [x2+distort_x2, y2-distort_y1], [x1-distort_x1, y2-distort_y2]], dtype=np.float32)

    # Define destination points for the transformation
    dst_pts = np.array(
        [[0, 0], [x2-x1, 0], [x2-x1, y2-y1], [0, y2-y1]], dtype=np.float32)

    # Get the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation to the input image
    transformed_img = cv.warpPerspective(img, M, (x2-x1, y2-y1))

    return transformed_img


output = cv.imread('3.jpg')
cv.imshow("Orig", output)

# Crop image:
# Detect the red dots
output = crop(output)


cv.imshow("cropped", output)

hsv_img = cv.cvtColor(output, cv.COLOR_BGR2HSV)

# Define the range of green color in HSV
lower_green = np.array([40, 50, 50])
upper_green = np.array([200, 255, 255])

# Create a mask for green color
green_mask = cv.inRange(hsv_img, lower_green, upper_green)

# Apply the mask to the original image
result = cv.bitwise_and(output, output, mask=green_mask)
gray = cv.cvtColor(cv.cvtColor(
    result, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 50, 100)
cv.imshow("edge", edges)


circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1,
                          minRadius=5, maxRadius=13, param2=15, minDist=25)


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
        cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


# Drawing the pixles to divide up the grid, development purposes only
dify = int(len(output)/8)+2
difx = int(len(output[0])/8)
const = 0
consty = 0
for i in range(8):
    for j in range(8):
        curr = gray[j * dify:(j+1) * dify, const+i *
                    difx:const+(i+1) * difx]
        cv.putText(output, text=str(np.round(np.mean(curr), 5)),
                   org=(const+i * difx, (j+1) * dify), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 0), thickness=1)

        cv.rectangle(output, (const+i * difx, j * dify-consty),
                     (const+(i+1) * difx, (j+1) * dify-consty), (0, 0, 255), 1)

# Start from the top.

board = []
for j in range(8):
    # For each row
    row = []
    for i in range(8):
        rect = [const+i * difx, j * dify, const+(i+1) * difx,
                (j+1) * dify]  # xMin, yMin, xMax, yMax
        print(rect)
        # check if a center of a circle is in that range
        isSomething = False
        for (x, y, r) in circles:
            if x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3]:
                isSomething = True
                row.append(0)
        if not isSomething:
            row.append(1)
    board.append(row)

pprint.pprint(board)


cv.imshow("Res", result)
cv.imshow("In", output)
cv.waitKey(0)
