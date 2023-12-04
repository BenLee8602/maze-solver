import cv2
import numpy as np

# extracts and processes maze contours from the input binary image
def get_maze_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ctimg = cv2.drawContours(image, contours, 0, (255, 255, 255), 5)
    ctimg = cv2.drawContours(ctimg, contours, 1, (0, 0, 0), 5)
    _, ctimg = cv2.threshold(ctimg, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((19, 19), np.uint8)
    return ctimg, kernel

# applies morphological operations to extract the paths within the maze
def get_maze_path(contours, kernel):
    dilation = cv2.dilate(contours, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return cv2.absdiff(dilation, erosion)

# builds the final output by applying a mask to the original image
def build_output(image, solution):
    b, g, r = cv2.split(image)
    mask_inv = cv2.bitwise_not(solution)
    b = cv2.bitwise_and(b, b, mask=mask_inv)
    g = cv2.bitwise_and(g, g, mask=mask_inv)
    return cv2.merge((b, g, r))
