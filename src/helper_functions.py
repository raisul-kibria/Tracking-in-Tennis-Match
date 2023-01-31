import numpy as np
import cv2 as cv
from typing import List
np.random.seed(123)

def write_to_video(frame: np.ndarray, name: str):
    """
    helper function to create video from output sequences.
    """
    height, width, layers = frame.shape
    size = (width, height)
    out = cv.VideoWriter('../data/result_' + name + ".mp4", cv.VideoWriter_fourcc(*'mp4v'), 15, size)
    return out

def create_background_model(source: str, sampling_prob = 0.7, save = True, name = None):
    """
    This function samples frames from the sequence of images and uses
    temporal median filter to create the background model. 
    Parameters:
        source (str): the path to the video file
        sampling_prob (float): probaility for each frame to be sampled
        save (bool): determines whether to save the baackground model as image
    Returns:
        np.ndarray: the background mask
    """
    cap = cv.VideoCapture(cv.samples.findFile(source))
    # List of sampled frames
    image_sequences = []
    while(1):
        p = np.random.uniform()
        ret, frame = cap.read()
        if not ret:
            break
        if p > sampling_prob:
            image_sequences.append(frame)

    # Background Model using median filter
    is_np = np.array(image_sequences)
    background_model = np.median(is_np, axis=0, keepdims=True).astype('uint8')
    bg_gray = cv.cvtColor(background_model[0], cv.COLOR_BGR2GRAY)
    if save:
        if name:
            cv.imwrite(f"../data/{name}_background_model.png", bg_gray)
        else:
            cv.imwrite(f"../data/background_model.png", bg_gray)
    return bg_gray

def remove_background(frame: np.ndarray, bg_mask: np.ndarray, morph_op: str, morph_kernel: List, thresh=30):
    """
    Function to create and apply the foreground mask from the background model
    on a frame
    Parameters:
        frame (np.ndarray): image of the frame to be processed
        bg_mask (np.ndarray): background model image
        morph_op (str): each character representing the operation in order
        morph_kernel (List of int): kernel sizes for the morhological operations
        thresh (int): background subtraction threshold
    Returns:
        np.ndarray: processed foreground mask
    """
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Background subtraction
    fg_mask = cv.absdiff(frame_gray, bg_mask)

    # Binarizing the foreground mask
    fg_mask[fg_mask > thresh] = 255
    fg_mask[fg_mask <= thresh] = 0

    # Morphological Processing
    assert len(morph_op) == len(morph_kernel), "Please make sure every morphological operation has corresponding kernel size defined"
    for i, c in enumerate(morph_op):
        kernel_size = (morph_kernel[i],morph_kernel[i])
        # Dilation: d
        if c == "d":
            fg_mask = cv.dilate(fg_mask, np.ones(kernel_size))
        # Erosion: e
        elif c == "e":
            fg_mask = cv.erode(fg_mask, np.ones(kernel_size))
        # Closing: c
        elif c == "c":
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size))
        # Opening: o
        elif c == "o":
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size))
        else:
            raise "Operation not defined"
    processed_frame = cv.bitwise_and(frame, frame, mask = fg_mask)
    return processed_frame

def draw_function(out, x_p1, y_p1, w_p1, h_p1, x_p2, y_p2, w_p2, h_p2, x_b, y_b, w_b, h_b, p1_score, p2_score, current):
    """
    helper function for drawing tracked objects and the scores
    """
    # Draw Player 1
    cv.putText(out, f'Player#1', (x_p1,y_p1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv.LINE_AA)
    cv.rectangle(out, (x_p1,y_p1), (x_p1+w_p1,y_p1+h_p1), (255, 0, 0), 2)

    # Draw Player 2
    cv.putText(out, f'Player#2', (x_p2, y_p2 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv.LINE_AA)
    cv.rectangle(out, (x_p2, y_p2), (x_p2+w_p2,y_p2+h_p2), (0, 0, 255), 2)

    # Highlight Current Player
    if current == "p1":
        cv.rectangle(out, (30,65), (500//2, 70), (255, 0, 0), -1)
    if current == "p2":
        cv.rectangle(out, (500//2,65), (500, 70), (0, 0, 255), -1)

    # Draw Ball
    cv.putText(out, f'Tennis ball', (x_b, y_b - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.rectangle(out, (x_b,y_b), (x_b+w_b,y_b+h_b), (0, 255, 0),2)

    # Scoreboard
    cv.rectangle(out, (30,20), (500, 65), (220,220,220), -1)
    cv.putText(out, f'Player#1 [{p1_score} - {p2_score}] Player#2', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv.LINE_AA)
    return out