"""
scoring_functions.py
Author: Lauren Okamoto
"""

import pickle
import numpy as np
import os
import math
from scipy.signal import find_peaks
from rule_based_programs.microprograms.dive_recognition_functions import *

### All functions (excluding helper functions) take "dive_data" as input, 
### which is the dictionary with all the information outputted by the AQA metaprogram 

############## HELPER FUNCTIONS ######################################
def rotation_direction(vector1, vector2, threshold=0.4):
    # Calculate the determinant to determine rotation direction
    determinant = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    mag1= np.linalg.norm(vector1)
    mag2= np.linalg.norm(vector2)
    norm_det = determinant/(mag1*mag2)
    if norm_det > threshold:
        # "counterclockwise"
        return 1
    elif norm_det < 0-threshold:
        # "clockwise"
        return -1
    else:
        # "not determinent"
        return 0

def find_angle(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = math.degrees(np.arccos(dot_product))
    return angle

#################################################################

def height_off_board_score(dive_data):
    above_board_indices = [i for i in range(0, len(dive_data['distance_from_board'])) if dive_data['above_boards'][i]==1]
    takeoff_indices = [i for i in range(0, len(dive_data['takeoff'])) if dive_data['takeoff'][i]==1]
    final_indices = []
    prev_board_end_coord = None
    for i in range(len(above_board_indices)):
        board_end_coord = dive_data['board_end_coords'][i]
        if board_end_coord is not None and board_end_coord[1] < 30:
            continue
        if board_end_coord is not None and prev_board_end_coord is not None and math.dist(board_end_coord, prev_board_end_coord) > 150:
            continue
        if above_board_indices[i] not in takeoff_indices:
            final_indices.append(above_board_indices[i])
        prev_board_end_coord = board_end_coord
    
    heights = []
    for i in range(len(final_indices)):
        pose_pred = dive_data['pose_pred'][final_indices[i]]
        board_end_coord = dive_data['board_end_coords'][final_indices[i]]
        if pose_pred is None or board_end_coord is None:
            continue
        pose_pred = pose_pred[0]
        min_height = float('inf')
        for j in range(len(pose_pred)):
            if board_end_coord[1] - pose_pred[j][1]< min_height:
                min_height = board_end_coord[1] - pose_pred[j][1] 
                if min_height < 0:
                    min_height = 0
        heights.append(min_height)
    if len(heights) == 0:
        return None, None
    max_scaled_height = max(heights) / get_scale_factor(dive_data)
    return max_scaled_height, final_indices[np.argmax(heights)]

def distance_from_board_score(dive_data):
    above_board_indices = [i for i in range(0, len(dive_data['distance_from_board'])) if dive_data['above_boards'][i]==1]
    takeoff_indices = [i for i in range(0, len(dive_data['takeoff'])) if dive_data['takeoff'][i]==1]
    final_indices = []
    for i in range(len(above_board_indices)):
        if above_board_indices[i] not in takeoff_indices:
            final_indices.append(above_board_indices[i])
    dists = np.array(dive_data['distance_from_board'])[final_indices]
    for i in range(len(dists)):
        if dists[i] is None:
            dists[i] = float('inf')
    min_scaled_dist = np.min(dists) / get_scale_factor(dive_data)
    too_close_threshold = 0.25
    if 'diveNum' in dive_data:
        if dive_data['diveNum'][0] == '4':
            too_far_threshold = 1.1
        if dive_data['diveNum'][0] == '1':
            too_far_threshold = 1.6
        if dive_data['diveNum'][0] == '2':
            too_far_threshold = 1.8
        if dive_data['diveNum'][0] == '3':
            too_far_threshold = 1.6
        if dive_data['diveNum'][0] == '5':
            too_far_threshold = 1.5
        if dive_data['diveNum'][0] == '6':
            too_far_threshold = 1.1
    else:
        too_far_threshold = 1.8
    # good distance
    if min_scaled_dist < too_far_threshold and min_scaled_dist > too_close_threshold:
        return 0, min_scaled_dist, final_indices[np.argmin(dists)]
    # too far
    if min_scaled_dist >= too_far_threshold:
        return 1, min_scaled_dist, final_indices[np.argmin(dists)]
    # too close
    if min_scaled_dist <= too_close_threshold:
        return -1, min_scaled_dist, final_indices[np.argmin(dists)]
    return min_scaled_dist

def knee_bend_score(dive_data):
    if find_position(dive_data) == 'tuck':
        return None, None
    knee_bends = []
    for i in range(len(dive_data['pose_pred'])):
        if dive_data['som'][i] == 0:
            continue
        pose_pred = dive_data['pose_pred'][i]
        if pose_pred is None:
            continue
        pose_pred = pose_pred[0]
        knee_to_ankle = [pose_pred[1][0] - pose_pred[0][0], 0-(pose_pred[1][1]-pose_pred[0][1])]
        knee_to_hip = [pose_pred[1][0] - pose_pred[2][0], 0-(pose_pred[1][1]-pose_pred[2][1])]
        knee_bend = find_angle(knee_to_ankle, knee_to_hip)
        knee_bends.append(knee_bend)
    if len(knee_bends) == 0:
        return None, None
    som_indices = [i for i in range(0, len(dive_data['som'])) if dive_data['som'][i]==1]
    som_avg_knee_bend = np.mean(knee_bends)
    return 180 - som_avg_knee_bend, som_indices

def position_tightness_score(dive_data):
    som_indices = [i for i in range(0, len(dive_data['som'])) if dive_data['som'][i]==1]
    twist_indices = [i for i in range(0, len(dive_data['som'])) if dive_data['twist'][i]==1]
    som_tightness = np.array(dive_data['position_tightness'])[som_indices]
    twist_tightness = 180 - np.array(dive_data['position_tightness'])[twist_indices]

    # Compute the area using the composite trapezoidal rule.
    som_tightness = np.array(list(filter(lambda item: item is not None and item < 90, som_tightness)))
    twist_tightness = np.array(list(filter(lambda item: item is not None and item < 90, twist_tightness)))
    if len(som_indices) == 0:
        som_avg = None
    else:
        som_avg = np.mean(som_tightness)
    if len(twist_indices)==0:
        return som_avg, None, som_indices, twist_indices
    twist_avg = np.mean(twist_tightness)
    if som_avg is not None:
        som_avg -= 15
    return som_avg, twist_avg, som_indices, twist_indices

def is_rotating_clockwise(dive_data):
    directions = []
    for i in range(1, len(dive_data['pose_pred'])):
        if dive_data['pose_pred'][i] is None or dive_data['pose_pred'][i-1] is None:
            continue
        if dive_data['on_boards'][i] == 0:
            prev_pose_pred_hip = dive_data['pose_pred'][i-1][0][3]
            curr_pose_pred_hip = dive_data['pose_pred'][i][0][3]
            prev_pose_pred_knee = dive_data['pose_pred'][i-1][0][4]
            curr_pose_pred_knee = dive_data['pose_pred'][i][0][4]   
            prev_hip_knee = [prev_pose_pred_knee[0] - prev_pose_pred_hip[0], 0-(prev_pose_pred_knee[1] - prev_pose_pred_hip[1])]
            curr_hip_knee = [curr_pose_pred_knee[0] - curr_pose_pred_hip[0], 0-(curr_pose_pred_knee[1] - curr_pose_pred_hip[1])]
            direction = rotation_direction(prev_hip_knee, curr_hip_knee, threshold=0)
            directions.append(direction)
    return np.sum(directions) < 0

def over_under_rotation_score(dive_data):
    entry_indices = [i for i in range(0, len(dive_data['entry'])) if dive_data['entry'][i]==1]
    over_under_rotation_error = np.array(dive_data['over_under_rotation'])[entry_indices]
    splashes = np.array(dive_data['splash'])[entry_indices]
    for i in range(len(over_under_rotation_error) - 1, -1, -1):
        if over_under_rotation_error[i] is None:
            continue
        else:
            # gets the second to last pose (assuming the last pose has incorrect pose estimation)
            index = i-2
            if index < 0:
                index = 0
            total_index = entry_indices[index]
            if splashes[index] is None and over_under_rotation_error[index] is not None:
                pose_pred = dive_data['pose_pred'][total_index][0]
                thorax_pelvis_vector = [pose_pred[1][0] - pose_pred[7][0], 0-(pose_pred[1][1]-pose_pred[7][1])]
                prev_pose_pred = dive_data['pose_pred'][total_index - 1]
                if prev_pose_pred is not None:
                    prev_pose_pred = prev_pose_pred[0]
                    prev_thorax_pelvis_vector = [prev_pose_pred[1][0] - prev_pose_pred[7][0], 0-(prev_pose_pred[1][1]-prev_pose_pred[7][1])]
                    rotation_speed = find_angle(thorax_pelvis_vector, prev_thorax_pelvis_vector)
                else:
                    rotation_speed = 10
                vector2 = [0, 1]
                if is_rotating_clockwise(dive_data):
                    # if under-rotated
                    if thorax_pelvis_vector[0] < 0:
                        avg_leg_torso = find_angle(thorax_pelvis_vector, vector2) - rotation_speed
                        
                    else:
                        avg_leg_torso = find_angle(thorax_pelvis_vector, vector2) + rotation_speed
                else:
                    # if over-rotated
                    if thorax_pelvis_vector[0] < 0:  
                        avg_leg_torso = find_angle(thorax_pelvis_vector, vector2) + rotation_speed
                    else:
                        avg_leg_torso = find_angle(thorax_pelvis_vector, vector2) - rotation_speed
                return np.abs(avg_leg_torso), entry_indices[index]
                break

def straightness_during_entry_score(dive_data):
    entry_indices = [i for i in range(0, len(dive_data['entry'])) if dive_data['entry'][i]==1]
    straightness_during_entry = np.array(dive_data['position_tightness'])[entry_indices]
    over_under_rotation = over_under_rotation_score(dive_data)
    if over_under_rotation is not None:
        frame = over_under_rotation[1]
        index = entry_indices.index(frame) - 1
        if index < 0:
            index = 0
        return 180-straightness_during_entry[index], [frame-1, frame, frame + 1]
    splashes = np.array(dive_data['splash'])[entry_indices]
    for i in range(len(straightness_during_entry) - 1, -1, -1):
        if i > 0 and (straightness_during_entry[i] is None or splashes[i] is not None):
            continue
        else:
            # gets the second to last pose (assuming the last pose has incorrect pose estimation)
            if straightness_during_entry[i] is not None:
                if 180-straightness_during_entry[i] > 130:
                    continue
                return 180-straightness_during_entry[i], entry_indices[i-1:i+2]
                break

def splash_score(dive_data):
    entry_indices = [i for i in range(0, len(dive_data['entry'])) if dive_data['entry'][i]==1]
    if len(entry_indices) == 0:
        return None
    splash_indices=[i for i in range(0, len(dive_data['splash'])) if dive_data['splash'][i] is not None]
    splashes = np.array(dive_data['splash'])[entry_indices]
    for i in range(len(splashes)):
        if splashes[i] is None:
            splashes[i] = 0
    splashes = splashes / get_scale_factor(dive_data)**2
    # area under curve
    area = np.trapz(splashes, dx=5) 
    return area, entry_indices[np.argmax(splashes)], splash_indices

# feet apart
def feet_apart_score(dive_data):
    takeoff_indices = [i for i in range(0, len(dive_data['takeoff'])) if dive_data['takeoff'][i]==1]
    non_takeoff_indices = [i for i in range(len(dive_data['takeoff'])) if (i not in takeoff_indices and dive_data['splash'][i] is None)]
    feet_apart_error = np.array(dive_data['feet_apart'])[non_takeoff_indices]
    for i in range(len(feet_apart_error)):
        if feet_apart_error[i] is None or math.isnan(feet_apart_error[i]):
            feet_apart_error[i] = 0
    peaks, _ = find_peaks(feet_apart_error, height=5)
    if len(peaks) >= 1:
        peak_indices = np.array(non_takeoff_indices)[peaks]
    else:
        peak_indices = []
    area = np.mean(feet_apart_error)

    return area, peak_indices

def find_position(dive_data):
    angles = []
    three_in_a_row = 0
    for i in range(1, len(dive_data['pose_pred'])):
        pose_pred = dive_data['pose_pred'][i]
        if pose_pred is None or dive_data['som'][i]==0:
            continue
        pose_pred = pose_pred[0]
        l_knee = pose_pred[4]
        l_ankle = pose_pred[5]
        l_hip = pose_pred[3]
        l_knee_ankle = [l_ankle[0] - l_knee[0], 0-(l_ankle[1] - l_knee[1])]
        l_knee_hip = [l_hip[0] - l_knee[0], 0-(l_hip[1] - l_knee[1])]
        angle = find_angle(l_knee_ankle, l_knee_hip)
        angles.append(angle)
        # print(angle)
        if angle < 70:
            three_in_a_row += 1
            if three_in_a_row >=3:
                return 'tuck'
        else:
            three_in_a_row =0
    return 'pike'

def get_position_from_diveNum(dive_data):
    diveNum = dive_data['diveNum']
    position_code = diveNum[-1]
    if position_code == 'a':
        return "straight"
    elif position_code == 'b':
        return "pike"
    elif position_code == 'c':
        return "tuck"
    elif position_code == 'd':
        return "free"
    else:
        return None

def get_all_report_scores(dive_data):
    with open('distribution_data.pkl', 'rb') as f:
        distribution_data = pickle.load(f)
     ## handstand and som_count##
    expected_som, handstand = som_counter_full_dive(dive_data)
    ## twist_count
    expected_twists = twist_counter_full_dive(dive_data)
    ## direction: front, back, reverse, inward
    expected_direction = get_direction(dive_data)
    dive_data['is_handstand'] = handstand
    dive_data['direction'] = expected_direction

    intermediate_scores = {}
    all_percentiles = []
    entry_indices = [i for i in range(0, len(dive_data['entry'])) if dive_data['entry'][i]==1]

    ### height off board ###
    if dive_data['is_handstand']:
        error_scores = distribution_data['armstand_height_off_board_scores']
    elif expected_twists >0:
        error_scores = distribution_data['twist_height_off_board_scores']
    elif dive_data['direction']=='front':
        error_scores = distribution_data['front_height_off_board_scores']
    elif dive_data['direction']=='back':
        error_scores = distribution_data['back_height_off_board_scores']
    elif dive_data['direction']=='reverse':
        error_scores = distribution_data['reverse_height_off_board_scores']
    elif dive_data['direction']=='inward':
        error_scores = distribution_data['inward_height_off_board_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['height_off_board'] = {}
    if dive_data['is_handstand']:
        intermediate_scores['height_off_board']['raw_score'] = None
        intermediate_scores['height_off_board']['frame_index'] = None
    else:
        intermediate_scores['height_off_board']['raw_score'] = height_off_board_score(dive_data)[0]
        intermediate_scores['height_off_board']['frame_index'] = height_off_board_score(dive_data)[1]
    err = intermediate_scores['height_off_board']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['height_off_board']['percentile'] = temp.index(err)/len(temp)
        all_percentiles.append(temp.index(err)/len(temp))
    else:
        intermediate_scores['height_off_board']['percentile'] = None

    ## distance from board ####
    error_scores = distribution_data['distance_from_board_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['distance_from_board'] = {}
    intermediate_scores['distance_from_board']['raw_score'] = distance_from_board_score(dive_data)[1]
    intermediate_scores['distance_from_board']['frame_index'] = distance_from_board_score(dive_data)[2]
    err = distance_from_board_score(dive_data)[0]
    if err is not None:
        if err == 1:     
            intermediate_scores['distance_from_board']['percentile'] = "safe, but too far from"
            intermediate_scores['distance_from_board']['score'] = 0.5

        elif err == 0:
            intermediate_scores['distance_from_board']['percentile'] = "a good distance from"
            intermediate_scores['distance_from_board']['score'] = 1
        else:
            intermediate_scores['distance_from_board']['percentile'] = "too close to"
            intermediate_scores['distance_from_board']['score'] = 0
        all_percentiles.append(intermediate_scores['distance_from_board']['score'])
    else:
        intermediate_scores['distance_from_board']['percentile'] = None
        intermediate_scores['distance_from_board']['score'] = None
        
    ### feet_apart_scores ###
    error_scores = distribution_data['feet_apart_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['feet_apart'] = {}
    intermediate_scores['feet_apart']['raw_score'] = feet_apart_score(dive_data)[0]
    intermediate_scores['feet_apart']['peaks'] = feet_apart_score(dive_data)[1]
    err = intermediate_scores['feet_apart']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['feet_apart']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))
    else:
        intermediate_scores['feet_apart']['percentile'] = None

    ### knee_bend_scores ###
    error_scores = distribution_data['knee_bend_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['knee_bend'] = {}
    intermediate_scores['knee_bend']['raw_score'] = knee_bend_score(dive_data)[0]
    intermediate_scores['knee_bend']['frame_indices'] = knee_bend_score(dive_data)[1]
    err = intermediate_scores['knee_bend']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['knee_bend']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))
    else:
        intermediate_scores['knee_bend']['percentile'] = None
        

    ### som_position_tightness_scores ###
    error_scores = distribution_data['som_position_tightness_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['som_position_tightness'] = {}
    position = find_position(dive_data)
    if position == 'tuck':
        intermediate_scores['som_position_tightness']['position'] = 'tuck'
    else:
        intermediate_scores['som_position_tightness']['position'] = 'pike'
    intermediate_scores['som_position_tightness']['raw_score'] = position_tightness_score(dive_data)[0]
    intermediate_scores['som_position_tightness']['frame_indices'] = position_tightness_score(dive_data)[2]
    err = intermediate_scores['som_position_tightness']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['som_position_tightness']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))
    else:
        intermediate_scores['som_position_tightness']['percentile'] = None
    
    ### twist_position_tightness_scores ###
    error_scores = distribution_data['twist_position_tightness_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['twist_position_tightness'] = {}
    intermediate_scores['twist_position_tightness']['raw_score'] = position_tightness_score(dive_data)[1]
    intermediate_scores['twist_position_tightness']['frame_indices'] = position_tightness_score(dive_data)[3]
    err = intermediate_scores['twist_position_tightness']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['twist_position_tightness']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))
    else:
        intermediate_scores['twist_position_tightness']['percentile'] = None
        
    ### over_under_rotation_scores ###
    error_scores = distribution_data['over_under_rotation_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['over_under_rotation'] = {}
    if over_under_rotation_score(dive_data) is not None:
        intermediate_scores['over_under_rotation']['raw_score'] = over_under_rotation_score(dive_data)[0]
        intermediate_scores['over_under_rotation']['frame_index'] = over_under_rotation_score(dive_data)[1]
    else:
        intermediate_scores['over_under_rotation']['raw_score'] = None
        intermediate_scores['over_under_rotation']['frame_index'] = None
    err = intermediate_scores['over_under_rotation']['raw_score']
    if err is not None:
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['over_under_rotation']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))

    else:
        intermediate_scores['over_under_rotation']['percentile'] = None
        
    ### splash_scores ###
    error_scores = distribution_data['splash_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['splash'] = {}
    intermediate_scores['splash']['raw_score'] = splash_score(dive_data)[0]
    intermediate_scores['splash']['maximum_index'] = splash_score(dive_data)[1]
    intermediate_scores['splash']['frame_indices'] = splash_score(dive_data)[2]

    err = intermediate_scores['splash']['raw_score']
    if err is not None:    
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['splash']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))

    else:
        intermediate_scores['splash']['percentile'] = None
        
    ### straightness_during_entry_scores ###
    error_scores = distribution_data['straightness_during_entry_scores']
    error_scores = list(filter(lambda item: item is not None, error_scores))
    intermediate_scores['straightness_during_entry'] = {}
    if straightness_during_entry_score(dive_data) is not None:
        intermediate_scores['straightness_during_entry']['raw_score'] = straightness_during_entry_score(dive_data)[0]
        intermediate_scores['straightness_during_entry']['frame_indices'] = straightness_during_entry_score(dive_data)[1]
    else:
        intermediate_scores['straightness_during_entry']['raw_score'] = None
        intermediate_scores['straightness_during_entry']['frame_index'] = None
        
    err = intermediate_scores['straightness_during_entry']['raw_score']
    if err is not None:    
        temp = error_scores
        temp.append(err)
        temp.sort()
        intermediate_scores['straightness_during_entry']['percentile'] = 1-temp.index(err)/len(temp)
        all_percentiles.append(1-temp.index(err)/len(temp))

    else:
        intermediate_scores['straightness_during_entry']['percentile'] = None
        
    ## overall score ###
    # Excellent:                   10 
    # Very Good:             8.5-9.5 
    # Good:                      7.0-8.0 
    # Satisfactory:           5.0-6.5 
    # Deficient:                2.5-4.5 
    # Unsatisfactory:   	0.5-2.0 
    # Completely failed:  	0  
    overall_score = np.mean(all_percentiles) * 10
    intermediate_scores['overall_score'] = {}
    intermediate_scores['overall_score']['raw_score'] = overall_score
    if overall_score == 10:
        intermediate_scores['overall_score']['description'] = 'excellent'
    elif overall_score >=8.5 and overall_score <10:
        intermediate_scores['overall_score']['description'] = 'very good'
    elif overall_score >=7 and overall_score <8.5:
        intermediate_scores['overall_score']['description'] = 'good'
    elif overall_score >=5 and overall_score <7:
        intermediate_scores['overall_score']['description'] = 'satisfactory'
    elif overall_score >=2.5 and overall_score <5:
        intermediate_scores['overall_score']['description'] = 'deficient'
    elif overall_score >0 and overall_score <2.5:
        intermediate_scores['overall_score']['description'] = 'unsatisfactory'
    else:
        intermediate_scores['overall_score']['description'] = 'completely failed'
    
    return intermediate_scores
