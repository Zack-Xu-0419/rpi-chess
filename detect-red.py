import cv2 as cv
import numpy as np
# import picamera

# # Get Image
# with picamera.PiCamera() as camera:
#     camera.resolution = (640, 480)
#     output = np.empty((640, 480, 3))
#     camera.capture(output, 'rgb')
#     camera.close()


def edge_det():
    output = cv.imread('5.jpg')

    hsv_img = cv.cvtColor(output, cv.COLOR_BGR2HSV)

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
            cv.circle(result, (int(i[0]), int(i[1])),
                      radius=10, color=(0, 255, 0))


cv.imshow("Res", result)
cv.imshow("In", output)
cv.waitKey(0)
