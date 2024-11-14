import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math

# return image_rotated np.array, bb_points_sorted_rot np.array[4x2], roi_image np.array (rotated image cropped), angle float
def rotate_boundingBox(image, bb_points_sorted, bb_width, bb_height, fix_horizontalBars_case=True, 
                       visualize_rotatedImage_boundingBox=False):
    # fix_horizontalBars -> to fix the horizontal bars case*

    # Rotate the image and the bounding box, such that the bounding box becomes perfectly aligned with the image axes
    image_rot, bb_points_sorted_rot, angle = _rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height)

    # If the barcode is rotated (i.e. the bars are perfectly horizontal), then we must perform a rotation of the image
    if fix_horizontalBars_case:
        image_rot, bb_points_sorted_rot, bb_width, bb_height = _fix_horizontalBars_case(image_rot, bb_points_sorted_rot, 
                                                                                        bb_width, bb_height)

    if visualize_rotatedImage_boundingBox:
        image_rot_bb = image_rot.copy()
        cv2.drawContours(image_rot_bb, [sort_bb_points_for_visualization(bb_points_sorted_rot)], -1, (0, 255, 0), 3)
        plt.figure()
        plt.imshow(image_rot_bb, 'gray')
        plt.title('Rotated image')

    # Gray-scale rotated image
    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)  
    # Crop the rotated image around the rotated bounding box: ROI image
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                             int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]

    return image_rot, bb_points_sorted_rot, roi_image, angle

# Rotate the image by the given angle with respect to the given centre
def _rotate_image(image, angle, center):
  # Rotation matrix
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  # Rotated image
  image_rot = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return image_rot, rot_mat

# return image_rot np.array, bb_points_sorted_rot np.array[4x2], angle float
def _rotate_image_boundingBox(image, bb_points_sorted, bb_width, bb_height):
    # rotate the image and the bounding box to perfectly align the bounding box with the image axes
    
    # First two bounding box verteces
    point1 = bb_points_sorted[0, :]
    point2 = bb_points_sorted[1, :]

    # Angle between the line connecting point1-point2 and the horixontal axis
    angle = math.degrees(math.atan((point2[1]-point1[1])/(point2[0]-point1[0])))
    
    # The angle is 0: the bounding box is already perfectly aligned with the image axes.
    # No rotation is perfomed.
    if abs(angle)<10**(-4):  
        image_rot, bb_points_sorted_rot = image, bb_points_sorted
    
    else:  # The angle is not 0: rotation is needed

        # Bounding box rotated
        bb_points_sorted_rot = np.array([point1,
                              [point1[0]+bb_width-1,point1[1]],
                              [point1[0],point1[1]+bb_height-1],
                              [point1[0]+bb_width-1,point1[1]+bb_height-1]], dtype='float32') 
        
        # Rotate the image, by angle `angle` and with respect to the centre `point1`
        image_rot, rot_mat = _rotate_image(image, angle=angle, center=point1)
    
    angle *= -1

    return image_rot, bb_points_sorted_rot, angle


# return image_rot_rot np.array, bb_points_sorted_rot_rot np.array[4x2], bb_width int, bb_height int
def _fix_horizontalBars_case(image_rot, bb_points_sorted_rot, bb_width, bb_height):
    # If necessary, fix the horizontal barcode bars problem to have the bars perfectly vertical instead of perfectly horizontal
    # image_rot -> image already rotated to have the bounding box perfectly aligned with the image axes

    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)  # Gray-scale rotated image
    roi_image = gray_rot[int(bb_points_sorted_rot[0][1]):int(bb_points_sorted_rot[0][1]+bb_height), 
                         int(bb_points_sorted_rot[0][0]):int(bb_points_sorted_rot[0][0]+bb_width)]


    # If the barcode is rotated the vertical gradient is bigger than the horizontal gradient
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(roi_image, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(roi_image, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    barcode_rotated = cv2.convertScaleAbs(gradY).sum()>cv2.convertScaleAbs(gradX).sum()

    if not barcode_rotated:
        return image_rot, bb_points_sorted_rot, bb_width, bb_height 

    else:
        # The bars are perfectly horizontal
        # A rotation is needed, implemented through a warping/homography.

        # Dimensions of the input image
        height, width = gray_rot.shape

        # Source points for computing the homography. These are the four verteces of the current image (rotated image).
        coordinates_source = np.array([[0, 0],
                                    [width-1, 0],
                                    [0, height-1],
                                    [width-1, height-1]], dtype='float32')

        # The width becomes the height and the height becomes the width.
        destination_height, destination_width = width, height

        # Corresponding destination points, for computing the homography.
        coordinates_destination = np.array([[destination_width-1, 0],
                                                [destination_width-1, destination_height-1],
                                                [0, 0],                                        
                                                [0, destination_height-1]], dtype='float32')

        # Computing the trasformation (rotation) homography
        H = cv2.getPerspectiveTransform(coordinates_source, coordinates_destination)

        # Applying the trasformation
        image_rot_rot = cv2.warpPerspective(image_rot, H, (destination_width, destination_height))

        # Applying the trasformation: we rotated the bounding box verteces
        bb_points_sorted_rot_rot = cv2.perspectiveTransform(bb_points_sorted_rot.reshape(-1,1,2),H)
        bb_points_sorted_rot_rot = bb_points_sorted_rot_rot[:,0,:]
        # Sort the verteces: upper-left -> upper-right -> lower-left -> lower-right
        bb_points_sorted_rot_rot = sort_bb_points(bb_points_sorted_rot_rot)  

        # Width and height of the new rotated image (the dimensions are swapped)
        bb_width, bb_height = bb_height, bb_width

        return image_rot_rot, bb_points_sorted_rot_rot, bb_width, bb_height
    
def sort_bb_points(bb_points):
    # Function which sorts the bounding box points: upper-left -> upper-right -> lower-left -> lower-right.

    min_width = bb_points[:,0].min()
    min_height = bb_points[:,1].min()
    max_width = bb_points[:,0].max()
    max_height = bb_points[:,1].max()
    def normalize(value, axis=0):
        if axis==0:  # Horizontal dimension
            return min_width if (value-min_width<max_width-value) \
                            else max_width
        elif axis==1:  # Vertical dimension
            return min_height if (value-min_height<max_height-value) \
                            else max_height
    bb_points_sorted = np.array(sorted([tuple(v) for v in bb_points], key=lambda t: (normalize(t[1], axis=1),
                                                                                                normalize(t[0], axis=0))))

    return bb_points_sorted


def sort_bb_points_for_visualization(bb_points_sorted):
    # Sort to plot
    bb_rot = bb_points_sorted.copy()
    bb_rot[2, :] = bb_points_sorted[3, :]
    bb_rot[3, :] = bb_points_sorted[2, :]
    return bb_rot.astype(int)