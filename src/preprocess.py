import cv2
import numpy as np


# converts input image to binary
def binary_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return image


# returns the coordinates of the maze corners
def detect_maze(image):
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = image.shape[0] * image.shape[1] // 8
    largest_quad = None

    for cnt in contours:
        # approximate polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # if quadrilateral
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > largest_area:
                largest_area = area
                largest_quad = approx
    
    if largest_quad is None:
        return None
    return np.float32([c[0] for c in largest_quad])


# perspective transforms the image so the maze is completely straight
def perspective_transform_maze(image, corners, dsize=400):
    if corners is None:
        return image
    dest = np.float32([
        [0, 0],
        [dsize - 1, 0],
        [dsize - 1, dsize - 1],
        [0, dsize - 1]
    ])
    matrix = cv2.getPerspectiveTransform(corners, dest)
    result = cv2.warpPerspective(image, matrix, (dsize, dsize))
    return result
