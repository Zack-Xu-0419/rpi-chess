import cv2
import numpy as np

img = cv2.imread("3.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

cv2.imshow("r", edges)
cv2.waitKey()