import cv2
import numpy as np


def main():
    # Read input image
    img = cv2.imread("1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    # Number of internal corners in the chessboard pattern
    pattern_size = (7, 7)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refine the corners' positions
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Get the perspective transformation matrix
        src_pts = corners.reshape(-1, 2)
        square_size = 50  # Size of one chessboard square
        pattern_points = np.zeros((np.prod(pattern_size), 2), np.float32)
        pattern_points[:, 0] = np.tile(
            np.arange(pattern_size[0]), pattern_size[1]) * square_size
        pattern_points[:, 1] = np.repeat(
            np.arange(pattern_size[1]), pattern_size[0]) * square_size
        M, _ = cv2.findHomography(src_pts, pattern_points)

        # Apply the perspective transformation to the input image
        board_size = (pattern_size[0] * square_size,
                      pattern_size[1] * square_size)
        transformed_img = cv2.warpPerspective(img, M, board_size)

        # Show the transformed image
        cv2.imshow("Transformed Image", transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Chessboard not found.")


if __name__ == "__main__":
    main()
