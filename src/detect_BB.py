import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

# return bb_points_sorted [4x2], bb_width int, bb_height int, threshold int
def detect_boundingBox(image, visualize_bounding_box=False):

    # Convert the image to grey scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute both the horizontal and vertical derivative -> Sobel filter
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # Blur the gradient image    
    blurred = cv2.blur(gradient, (9, 9))

    # Threshold the gradient image -> Otsu's algorithm
    threshold, gray_thresholded = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Closing operator
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(gray_thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Opening operator
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(opened, kernel, iterations=3)
    
    # Find the bounding box
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)


    # to test the function
    if visualize_bounding_box:
        # Draw a bounding box around the detected barcode
        image_bb = image.copy()
        cv2.drawContours(image_bb, [box], -1, (0, 255, 0), 3)
        plt.figure()
        plt.imshow(image_bb, 'gray')
        plt.title('Bounding box')
    
    # Sorting the points of the bounding box: up-left, up-right, bottom-left, bottom-right
    bb_points_sorted = sort_bb_points(box.astype('float32'))
    
    # Height of the bounding box
    bb_height = int(max([dist(bb_points_sorted[0],bb_points_sorted[2]),
                             dist(bb_points_sorted[1],bb_points_sorted[3])]))
    # Width of the bounding box
    bb_width = int(max([dist(bb_points_sorted[0],bb_points_sorted[1]),
                             dist(bb_points_sorted[2],bb_points_sorted[3])]))


    return bb_points_sorted, bb_width, bb_height, threshold

def sort_bb_points(bb_points):
    min_width = bb_points[:,0].min()
    max_width = bb_points[:,0].max()
    min_height = bb_points[:,1].min()
    max_height = bb_points[:,1].max()
    def normalize(value, axis=0):
        if axis==0:  # Horizontal dimension
            return min_width if (value-min_width<max_width-value) else max_width
        elif axis==1:  # Vertical dimension
            return min_height if (value-min_height<max_height-value) else max_height
    bb_points_sorted = np.array(sorted([tuple(v) for v in bb_points], key=lambda t: (normalize(t[1], axis=1), normalize(t[0], axis=0))))
    return bb_points_sorted

# Compute the width and height of the bounding box.
def dist(point1, point2):
    return np.sqrt(np.sum((point1-point2)**2))