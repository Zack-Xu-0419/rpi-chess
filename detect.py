import cv2 as cv
import numpy as np
# import picamera

# # Get Image
# with picamera.PiCamera() as camera:
#     camera.resolution = (640, 480)
#     output = np.empty((640, 480, 3))
#     camera.capture(output, 'rgb')
#     camera.close()

output = cv.imread('2.jpg')

hsv_img = cv.cvtColor(output, cv.COLOR_BGR2HSV)

# Define the range of green color in HSV
lower_green = np.array([60, 100, 100])
upper_green = np.array([180, 255, 255])

# Create a mask for green color
green_mask = cv.inRange(hsv_img, lower_green, upper_green)

# Apply the mask to the original image
result = cv.bitwise_and(output, output, mask=green_mask)

edges = cv.Canny(cv.cvtColor(cv.cvtColor(
    result, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY), 50, 100)
cv.imshow("edge", edges)

circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1,
                          minRadius=10, maxRadius=30, param2=20, minDist=50)
print(len(circles))
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        print(x, y, r)
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv.circle(output, (x, y), r, (0, 255, 0), 4)
        cv.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv.imshow("Res", result)
cv.imshow("In", output)
cv.waitKey(0)
