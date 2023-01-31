import numpy as np
import cv2 as cv
from typing import List, Tuple

def tracking_object_setup(processed_frame: np.ndarray, track_window: Tuple):
    """
    function to initialize the parameters for object tracking via Mean shift
    Parameters:
        processed_frame (np.ndarray): frame with only the foreground objects
        track_window (list of int): coordinates of the target object
    Returns:
        np.ndarray: an array of histogram points of dtype float32.
        term_crit: termination criteria flag
    """
    (x,y,w,h) = track_window
    # set up the ROI for tracking
    roi = processed_frame[y:y+h, x:x+w]
    mask = cv.inRange(roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv.calcHist([roi], [0], mask, [180], [0,180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
    return roi_hist, term_crit

def track_using_meanshift(processed_frame: np.ndarray, roi_hist: np.ndarray, track_window: List, term_crit):
    """
    function that applies the mean shift algorithm
    Returns:
        Tuple: cooridnates of the tracked object
    """
    dst = cv.calcBackProject([processed_frame], [0], roi_hist, [0,180], 1)
    # apply meanshift to get the new location
    _, track_window = cv.meanShift(dst, track_window, term_crit)
    return track_window

def largest_contours(processed_frame: np.ndarray, max_rad=10000, debug=True):
    """
    function to find the largest contours corresponding to the players
    Parameters:
        min_rad (int): minimum area for a contour to be considered
        debug (bool): determines whether to show the frame
    Returns:
        np.ndarray: new position of the ball
        int: radius of the ball
    """
    # Find contours
    processed = cv.cvtColor(processed_frame, cv.COLOR_BGR2GRAY)
    processed[processed > 1] = 255
    contours, _ = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        cv.imshow("Frame for player detection", processed)

    contour_radius = []
    for c in contours:
        area = cv.contourArea(c)
        # if area > max_rad:
        #     area = 0
        contour_radius.append(area)
    
    sort_idx = np.argsort(contour_radius)
    print(np.array(contour_radius)[sort_idx])
    track_window_p1 = cv.boundingRect(contours[sort_idx[-1]])
    track_window_p2 = cv.boundingRect(contours[sort_idx[-2]])
    return track_window_p1, track_window_p2

def detect(processed_frame: np.ndarray, mask: np.ndarray, p_x: int, p_y: int, thresh_dist_min=5, thresh_dist_max=150, debug=False):
    """
    function to track the position of the ball from its corresponding contour
    Parameters:
        mask (np.ndarray): mask to remove the sides of the court
        p_x, p_y: last position of the ball
        thresh_dist_min (int): minimum distance the contour must have moved
        thresh_dist_max (int): maximum distance the contour can move
        debug (bool): determines whether to show the frame for ball tracking
    Returns:
        np.ndarray: new position of the ball
        int: radius of the ball
    """
    # Find contours
    processed = cv.bitwise_and(processed_frame, processed_frame, mask = mask)
    processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        cv.imshow("Frame for ball tracking", processed)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 2
    max_radius_thresh= 20

    smallest_contour = []
    smallest_dist = np.Inf

    for c in contours:
        (x, y), radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        # Take only the valid circle
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            dist = np.linalg.norm(np.array([p_x, p_y]) - np.array((x,y)))
            # contour is selected based on minimal movement and smallest radius
            score = (dist + radius)/2
            if score < smallest_dist and score > thresh_dist_min:
                smallest_dist = score
                smallest_contour = c

    # If the score is not over acceptable threshold, it is rejected
    if smallest_dist > thresh_dist_max:
        return None, None
    (x, y), radius = cv.minEnclosingCircle(smallest_contour)
    x = x - radius/2
    y = y - radius/2
    return np.array([[x], [y]]), int(radius)