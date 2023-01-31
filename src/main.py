import numpy as np
import cv2 as cv
import os
from helper_functions import create_background_model, remove_background, draw_function, write_to_video
from tracking_utils import tracking_object_setup, track_using_meanshift, detect, largest_contours

# FLAG
SHOW_RESULTS = True

source = '../data/video_cut.mp4'
cap = cv.VideoCapture(cv.samples.findFile(source))

if source == '../data/video_cut.mp4':
    thresh = 35
    p = 0.6
    bg_mask = cv.imread("../data/base_frame_mask_video_cut.png", 0)
    y_p1, h_p1, x_p1, w_p1 = 727, 971-727, 424, 591-424
    y_p2, h_p2, x_p2, w_p2 = 156, 303-156, 1015, 1105-1015
    x_b, y_b, w_b, h_b = 1049, 303, 20, 20
    morph_op = 'edc'
    morph_kernel = [3, 17, 15]
    turn_height = 80
    dist_crit = 150

elif source == '../data/video_input8.mp4':
    thresh = 30
    p = 0.5
    bg_mask = cv.imread("../data/base_frame_mask_video_input8.png", 0)
    y_p1,h_p1,x_p1,w_p1 = 722, 962-722, 598, 752-598
    y_p2,h_p2,x_p2,w_p2 = 82, 172-82, 831, 896-831
    x_b,y_b,w_b,h_b = 687, 588, 20, 20
    morph_op = 'dec'
    morph_kernel = [11, 5, 15]
    turn_height = 80
    dist_crit = 150

if os.path.isfile(f'../data/{source.split("/")[-1].split(".")[0]}_background_model.png'):
    background_model = cv.imread(f'../data/{source.split("/")[-1].split(".")[0]}_background_model.png', 0)
else:
    background_model = create_background_model(source, sampling_prob = p, save = True, name = source.split("/")[-1].split(".")[0])

p1_dists = []
p2_dists = []
p1_score, p2_score, p1_dist, p2_dist = 0, 0, 0, 0
p1_flag = True
p2_flag = True
hitby = ""
ball_track_meanshift = False
idx = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    result = frame.copy()
    processed_frame = remove_background(frame.copy(), background_model, morph_op=morph_op, morph_kernel=morph_kernel, thresh=thresh)
    # Setup objects in the first frame
    if idx == 0:
        video_writer = write_to_video(frame, name = source.split("/")[-1].split(".")[0])
        track_window_p1 = (x_p1, y_p1, w_p1, h_p1)
        track_window_p2 = (x_p2, y_p2, w_p2, h_p2)
        roi_hist_p1, term_crit_p1 = tracking_object_setup(processed_frame, track_window_p1)
        roi_hist_p2, term_crit_p2 = tracking_object_setup(processed_frame, track_window_p2)

        if ball_track_meanshift:        
            track_window_b = (x_b, y_b, w_b, h_b)
            roi_hist_b, term_crit_b = tracking_object_setup(processed_frame, track_window_b)
        
        # Player turn
        p1_dist = np.linalg.norm(np.array((x_p1+w_p1//2,y_p1+h_p1//2)) - np.array((x_b+w_b//2,y_b+h_b//2)))
        p2_dist = np.linalg.norm(np.array((x_p2+w_p2//2,y_p2+h_p2//2)) - np.array((x_b+w_b//2,y_b+h_b//2)))
        if p1_dist > p2_dist:
            current_player = "p1"
        else:
            current_player = "p2"
    else:
        # Player 1
        track_window_p1 = track_using_meanshift(processed_frame, roi_hist_p1, track_window_p1, term_crit_p1)
        x_p1,y_p1,w_p1,h_p1 = track_window_p1

        # Player 2
        track_window_p2 = track_using_meanshift(processed_frame, roi_hist_p2, track_window_p2, term_crit_p2)
        x_p2,y_p2,w_p2,h_p2 = track_window_p2

        # Ball
        if ball_track_meanshift:  
            track_window_b = track_using_meanshift(processed_frame, roi_hist_b, track_window_b, term_crit_b)
            x_b,y_b,w_b,h_b = track_window_b
        else:
            centers, r = detect(processed_frame, bg_mask, x_b, y_b)
            if r:
                x_b = int(centers[0][0])
                y_b = int(centers[1][0])
                (w_b, h_b) = (r, r)

    out = draw_function(frame.copy(), x_p1, y_p1, w_p1, h_p1, x_p2, y_p2, w_p2, h_p2, x_b, y_b, w_b, h_b, p1_score, p2_score)

    # Determine turn of players
    # ... 1. Determine distance to ball for the frame
    p1_dist_temp = np.linalg.norm(np.array((x_p1+w_p1//2,y_p1+h_p1//2)) - np.array((x_b+w_b//2,y_b+h_b//2)))
    p2_dist_temp = np.linalg.norm(np.array((x_p2+w_p2//2,y_p2+h_p2//2)) - np.array((x_b+w_b//2,y_b+h_b//2)))

    # ... 2. Keep track of the distances
    p1_dists.append(p1_dist - p1_dist_temp)
    p2_dists.append(p2_dist - p2_dist_temp)
    p1_dist = p1_dist_temp
    p2_dist = p2_dist_temp

    # ... 3. Active player is determined by accumulated distance towards the ball
    if sum(p1_dists) > sum(p2_dists) and current_player == "p2":
        current_player = "p1"
    else:
        current_player = "p2"
    
    # ... 4. Reset distances when ball is hit
    if current_player == "p1":
        if np.abs(y_p1 - y_b) < turn_height and p1_dist < dist_crit:
            hitby = "p1"
            p1_dists = []
            p2_dists = []
            if p1_flag:
                p1_score += 1
                p1_flag = False
                p2_flag = True
        # Loser criteria: active player unable to reach the ball when in same vertical level
        elif p1_dist > dist_crit:
            loser = "p1"

    if current_player == "p2":
        if np.abs(y_p2 - y_b) < turn_height and p2_dist < dist_crit:
            hitby = "p2"
            p1_dists = []
            p2_dists = []
            if p2_flag:
                p2_score += 1
                p2_flag = False
                p1_flag = True
        elif p1_dist > dist_crit:
            loser = "p2"

    
    video_writer.write(out)
    if SHOW_RESULTS:
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        cv.imshow(f'Frame', out)
        cv.imshow(f'Frame2', processed_frame)
    idx+=1

# Display final results
if loser == "p1":
    sc_color = (68, 56, 217)
if loser == "p2":
    sc_color = (255, 170, 18)
cv.rectangle(result, (30,20), (470, 65), sc_color, -1)
cv.putText(result, f'{"Player 1" if loser =="p1" else "Player2"} loses', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv.LINE_AA)
for i in range(30):
    video_writer.write(result)
if SHOW_RESULTS:
    cv.imshow(f'Frame', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
video_writer.release()
cap.release()
