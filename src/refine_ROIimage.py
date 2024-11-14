import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def refine_ROIimage(roi_image, image_rot, bb_points_sorted_rot, compute_barcode_structure_algorithm=1, threshold=None,
                       fix_wrongBar_case=True, outlier_detection_level=0.02, visualize_barcode_structure=False, 
                       visualize_refinedRoi_withQuantities=False, visualize_refinedRoi=False):
    """Refine the ROI image containing the barcode.
    Refinement: 10*X pixels before the first barcode bar and after the last barcode bar, where X is the minimum width of a bar.
    The height of the refined ROI image is equal to the minimum height of a barcode bar.

    The reference system is the current ROI image.
    - X : minimum width of a bar.
    - min_half_height_up : minimum half height of a bar from the middle of the ROI image upward.
    - min_half_height_down : minimum half height of a bar from the middle of the ROI image downward.
    - height : height of the barcode = min_half_height_up+min_half_height_down+1.
    - first_bar_x : horixontal coordinate of the first pixel of the first bar.
    - last_bar_x : horixontal coordinate of the last pixel of the last bar.
    - For each barcode bar:
        * Horixontal coordinate of the first pixel.
        * Width.
        * Height.
        * Half height from the middle of the ROI image upward.
        * Half height from the middle of the ROI image downward.

        
    Sometimes something which is not a barcode bar is detected as a bar.
    This situation can be fixed, by pruning from the barcode structure bars which are outliers with respect to the others.
    To minimize the possibility to wrongly prune an actual true bar, max one bar is pruned.

    Returns:
    roi_image_ref -> np.array -> ROI image after the refinement.
    bb_points_sorted_rot_ref -> np.array [4x2]
    barcode_structure_dict -> dict:
        - X : minimum width of a bar.
        - min_half_height_up : minimum half height of a bar from the middle of the ROI image upward.
        - min_half_height_down : minimum half height of a bar from the middle of the ROI image downward.
        - height : height of the barcode = min_half_height_up+min_half_height_down+1.
        - first_bar_x : horixontal coordinate of the first pixel of the first bar.
        - last_bar_x : horixontal coordinate of the last pixel of the last bar.
        - bars_start : list contaning, for each bar, the horixontal coordinate of the first pixel of that bar.
        - bars_width : list contaning, for each bar, the width of that bar.
        - bars_height : list contaning, for each bar, the height of that bar.
        - bars_halfHeightUp :  list contaning, for each bar, the half height from the middle of the ROI image upward of that bar.
        - bars_halfHeightDown :  list contaning, for each bar, the half height from the middle of the ROI image downward of that bar.
    """

    barcode_localStructure_dict = _compute_barcode_structure(roi_image, threshold=threshold, 
                                                            algorithm=compute_barcode_structure_algorithm)

    # Pruning the wrong bar from the barcode structure
    if fix_wrongBar_case:
        _fix_wrong_bar(barcode_localStructure_dict, level=outlier_detection_level) 

    # Compute the global barcode quantities, and create the final barcode structure dict
    barcode_structure_dict = barcode_localStructure_dict.copy()
    barcode_structure_dict['first_bar_x'] =  min(barcode_structure_dict['bars_start']) 
    barcode_structure_dict['last_bar_x'] =  max([s+w for s,w in zip(barcode_structure_dict['bars_start'],
                                                                    barcode_structure_dict['bars_width'])])-1 
    barcode_structure_dict['X'] =  min(barcode_structure_dict['bars_width'])  
    barcode_structure_dict['min_half_height_up'] =  min(barcode_structure_dict['bars_halfHeightUp'])   
    barcode_structure_dict['min_half_height_down'] =  min(barcode_structure_dict['bars_halfHeightDown'])
    barcode_structure_dict['height'] = barcode_structure_dict['min_half_height_up'] + barcode_structure_dict['min_half_height_down'] + 1

    if visualize_barcode_structure: 
        plot_barcode_Structure(roi_image, barcode_structure_dict)

    # Refine the ROI image, and the bounding box coordinates
    roi_image_ref, bb_points_sorted_rot_ref = _refine_roi(roi_image, image_rot, bb_points_sorted_rot, barcode_structure_dict, 
                                                         visualize_refinedRoi_withQuantities=visualize_refinedRoi_withQuantities, 
                                                         visualize_refinedRoi=visualize_refinedRoi)

    return roi_image_ref, bb_points_sorted_rot_ref, barcode_structure_dict


def plot_barcode_Structure(roi_image, barcode_structure_dict):
    bb_height = roi_image.shape[0]
    half_height = math.ceil(bb_height/2)

    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = (barcode_structure_dict['bars_start'], 
                                                                     barcode_structure_dict['bars_width'],
                                                                     barcode_structure_dict['bars_halfHeightUp'], 
                                                                     barcode_structure_dict['bars_halfHeightDown'])

    plt.figure(figsize=(10,10))
    roi_image_show = roi_image.copy()
    roi_image_show = cv2.cvtColor(roi_image_show, cv2.COLOR_GRAY2RGB ) 
    n_bars = len(bars_start)
    for b in range(n_bars):
        roi_image_show[[half_height-bars_halfHeightUp[b]-1,half_height+bars_halfHeightDown[b]-1],
                        bars_start[b]:bars_start[b]+bars_width[b],:] = np.array([255,0,0])
        roi_image_show[half_height-bars_halfHeightUp[b]-1:half_height+bars_halfHeightDown[b]-1+1,
                       [bars_start[b],bars_start[b]+bars_width[b]-1],:] = np.array([255,0,0])

    plt.imshow(roi_image_show)
    plt.title('Detailed Barcode')
    plt.show() 



def _compute_barcode_structure(roi_image, threshold=None, algorithm=1):
    # Theìreshold the ROI image, either by the given threshold or by using Otsu's
    if threshold is None:
        threshold ,ROI_thresh = cv2.threshold(roi_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        threshold, ROI_thresh = cv2.threshold(roi_image, threshold,255,cv2.THRESH_BINARY)

    if algorithm==1:
        algorithm_function = _algorithm1
    else:
        raise ValueError(f'Invalid algorithm {algorithm}')

    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = algorithm_function(ROI_thresh)  

    # Create the dictionary
    barcode_localStructure_dict = {}
    barcode_localStructure_dict['bars_start'] =  bars_start 
    barcode_localStructure_dict['bars_width'] =  bars_width   
    barcode_localStructure_dict['bars_height'] = [bars_halfHeightUp[i]+bars_halfHeightDown[i]+1 for i in range(len(bars_halfHeightUp))] 
    barcode_localStructure_dict['bars_halfHeightUp'] =  bars_halfHeightUp   
    barcode_localStructure_dict['bars_halfHeightDown'] =  bars_halfHeightDown  
    
    return barcode_localStructure_dict



# return float or None -> element of v most outlier value or none -> v is the vector monodimensional
def _find_outliers(v, level=0.02):
    q1, q3 = tuple(np.quantile(v, [level,1-level]))
    IQR = q3-q1
    outliers_mask = np.logical_or(v>q3+IQR, v<q1-IQR)
    if v[outliers_mask].size==0:
        return None
    return np.argmax([v[i]-q3-IQR if (outliers_mask[i] and v[i]-q3-IQR>0) else abs(v[i]-q1+IQR) if outliers_mask[i] else 0  for i in range(len(v))])


def _fix_wrong_bar(barcode_localStructure_dict, level=0.02):
    """Prune a possible wrongly-detected bar from the current barcode structure.
    Procedure:
    - Detect the outliers bars with respect to the bars height. 
    - Detect the outliers with respect to the bars area.
    - If the outlier bar with respect to the height and the outlier bar with respect to the area are different, end the 
      algorithm (no bar is pruned).
      Otherwise, the wrong bar is exactly that bar which is the most outlier bar both for the height and for the area.
    """
    bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown = (barcode_localStructure_dict['bars_start'], 
                                                                     barcode_localStructure_dict['bars_width'],
                                                                     barcode_localStructure_dict['bars_halfHeightUp'],
                                                                     barcode_localStructure_dict['bars_halfHeightDown'] ) 

    # Number of barcode bars
    n_bars = len(bars_start)
    # Height of each barcode bar
    bars_height = np.array([bars_halfHeightUp[i]+bars_halfHeightDown[i]+1 for i in range(n_bars)])
    # Area of each barcode bar
    bars_area = np.array([bars_height[i]+bars_width[i] for i in range(n_bars)])

    # Bar which is the most outlier bar with respect to the height
    wrong_bars_height_index = _find_outliers(bars_height, level=level)    
    
    # Bar which is the most outlier bar with respect to the area
    wrong_bars_area_index = _find_outliers(bars_area, level=level)
    
    if wrong_bars_area_index is None or wrong_bars_height_index is None or wrong_bars_area_index!=wrong_bars_height_index:
        return None


    wrong_bar_index = wrong_bars_area_index

    # Delete the wrong bar from the given input dictionary
    if wrong_bar_index is not None:  
        del bars_start[wrong_bar_index]
        del bars_width[wrong_bar_index]
        del bars_halfHeightUp[wrong_bar_index]
        del bars_halfHeightDown[wrong_bar_index]
                      


# return roi_image_ref np.array, bb_points_sorted_rot_ref np.array
def _refine_roi(roi_image, image_rot, bb_points_sorted_rot, barcode_structure_dict, visualize_refinedRoi_withQuantities=False,
               visualize_refinedRoi=False): 

    gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)

    bb_height, bb_width = roi_image.shape
                  
    first_bar_x, last_bar_x, X, min_half_height_up, min_half_height_down = (barcode_structure_dict['first_bar_x'], 
                                                                     barcode_structure_dict['last_bar_x'],
                                                                     barcode_structure_dict['X'],
                                                                     barcode_structure_dict['min_half_height_up'],
                                                                     barcode_structure_dict['min_half_height_down'] )

    half_height = math.ceil(bb_height/2)

    # Refinement of the bounding box to visualize
    bb_points_sorted_rot_ref = bb_points_sorted_rot.copy()
    bb_points_sorted_rot_ref[[0,2],0] = bb_points_sorted_rot[[0,2],0] - (10*X-first_bar_x) 
    bb_points_sorted_rot_ref[[1,3],0] = bb_points_sorted_rot[[1,3],0] + (10*X-(bb_width-last_bar_x-1))

    if visualize_refinedRoi_withQuantities:
        roi_image_ref = gray_rot[int(bb_points_sorted_rot_ref[0][1]):int(bb_points_sorted_rot_ref[2][1])+1, 
                                    int(bb_points_sorted_rot_ref[0][0]):int(bb_points_sorted_rot_ref[1][0])+1].copy()
        bb_width_ref  = roi_image_ref.shape[1]
        plt.figure()
        plt.imshow(roi_image_ref, 'gray')
        plt.axvline(10*X, c='orange', label='10*X')
        plt.axvline(bb_width_ref-10*X-1, c='red', label='-10*X')
        plt.axhline(half_height-min_half_height_up-1, c='green', label='X top')
        plt.axhline(half_height+min_half_height_down-1, c='blue', label='X down')
        plt.title('Refined ROI con Numeri')
        plt.legend()

    # Conclude the refinement of the bounding box and of the roi image. Refinement also along the height.
    bb_points_sorted_rot_ref[[0,1],1] = bb_points_sorted_rot[[0,1],1] + half_height - 1 - min_half_height_up 
    bb_points_sorted_rot_ref[[2,3],1] = bb_points_sorted_rot[[0,1],1] + half_height - 1 + min_half_height_down 
    roi_image_ref = gray_rot[int(bb_points_sorted_rot_ref[0][1]):int(bb_points_sorted_rot_ref[2][1])+1, 
                                int(bb_points_sorted_rot_ref[0][0]):int(bb_points_sorted_rot_ref[1][0])+1].copy()

    if visualize_refinedRoi: 
        plt.figure()
        plt.imshow(roi_image_ref, 'gray')
        plt.title('Refined ROI')

    return roi_image_ref, bb_points_sorted_rot_ref




# Algoritm1

def _algorithm1(ROI_thresh):
    bb_height, bb_width = ROI_thresh.shape

    # initializations
    half_height = math.ceil(bb_height/2)
    half_height_index = half_height-1

    bars_start = []
    bars_width = []
    bars_halfHeightUp = []
    bars_halfHeightDown = []

    i = 0  # Index for iterating over the pixels

    # scan each pixel along the horizontal line in the exact middle of the ROI image
    while i<bb_width:

        # White pixel: go to the next pixel
        if ROI_thresh[half_height_index, i]==255:
            i += 1
            continue

        # Black pixel
        # 'i' is the first pixel in this current barcode bar

        # Width of this current bar
        X_curr = 1    
        # Index representing the last pixel in this current bar
        i_end = i+1

        # go right, till finding a white pixel -> compute the width of this current bar.
        while ROI_thresh[half_height_index, i_end]==0:
            X_curr += 1
            i_end += 1

        # Now we search upward and downward along the vertical line 'i_med'.
        i_med = int((i+i_end)/2)
        j_up = half_height_index-1
        j_down = half_height_index+1
        half_height_up_curr = 0
        half_height_down_curr = 0

        # Cycle, in which we go upward and downard at the same time, for computing `half_height_up_curr` and 
        # `half_height_down_curr`
        up_reached = j_up<0 or (ROI_thresh[j_up, i_med]==255 and  ROI_thresh[j_up, i_med-1]==255 and ROI_thresh[j_up, i_med+1]==255)
        down_reached = j_down<0 or (ROI_thresh[j_down, i_med]==255 and  ROI_thresh[j_down, i_med-1]==255 and ROI_thresh[j_down, i_med+1]==255)
        while not up_reached or not down_reached:
            if not up_reached:
                j_up -= 1
                half_height_up_curr += 1
            if not down_reached:
                j_down += 1
                half_height_down_curr += 1
            up_reached = j_up<0 or (ROI_thresh[j_up, i_med]==255 and  ROI_thresh[j_up, i_med-1]==255 and ROI_thresh[j_up, i_med+1]==255)
            down_reached = j_down>=bb_height or (ROI_thresh[j_down, i_med]==255 and  ROI_thresh[j_down, i_med-1]==255 and ROI_thresh[j_down, i_med+1]==255)

        bars_start.append(i)
        bars_width.append(X_curr)
        bars_halfHeightUp.append(half_height_up_curr)
        bars_halfHeightDown.append(half_height_down_curr)

        # We update `i`: we pass to the white pixel right after the current bar
        i = i_end
    
    return bars_start, bars_width, bars_halfHeightUp, bars_halfHeightDown

