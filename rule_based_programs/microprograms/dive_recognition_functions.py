"""
dive_recognition_functions.py
Author: Lauren Okamoto
"""

import pickle
import numpy as np
import os
import math
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def get_scale_factor(dive_data):
    # distance between thorax and pelvis
    distances = []
    for pose_pred in dive_data['pose_pred']:
        if pose_pred is not None:
            distances.append(math.dist(pose_pred[0][6], pose_pred[0][7]))
    distances.sort()
    return np.median(distances)

def find_angle(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = math.degrees(np.arccos(dot_product))
    return angle

def is_back_facing(dive_data, board_side):
    directions = []
    for i in range(len(dive_data['pose_pred'])):
        pose_pred = dive_data['pose_pred'][i]
        if pose_pred is None or dive_data['above_boards'][i] == 0:
            continue
        pose_pred = pose_pred[0]
        
        ## left knee bend ###
        l_knee = pose_pred[4]
        l_ankle = pose_pred[5]
        l_hip = pose_pred[3]
        l_knee_ankle = [l_ankle[0] - l_knee[0], 0-(l_ankle[1] - l_knee[1])]
        l_knee_hip = [l_hip[0] - l_knee[0], 0-(l_hip[1] - l_knee[1])]
        l_direction = rotation_direction(l_knee_hip, l_knee_ankle)
        
        ## right knee bend ###
        r_knee = pose_pred[1]
        r_ankle = pose_pred[0]
        r_hip = pose_pred[2]
        r_knee_ankle = [r_ankle[0] - r_knee[0], 0-(r_ankle[1] - r_knee[1])]
        r_knee_hip = [r_hip[0] - r_knee[0], 0-(r_hip[1] - r_knee[1])]
        r_direction = rotation_direction(r_knee_hip, r_knee_ankle)
        if l_direction == r_direction and l_direction != 0 and board_side == 'left':
            # we're looking for more clockwise
            return l_direction < 0
        elif l_direction == r_direction and l_direction != 0:
            # we're looking for more counterclockwise
            return l_direction > 0
    return False

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

# returns None if either pose_pred or board_end_coord is None
# returns True if diver is on board, returns False if diver is off board
def detect_on_board(board_end_coord, board_side, pose_pred, handstand):
    if pose_pred is None:
        return
    if board_end_coord is None:
        return
    if board_side == 'left':
        # if right of board end
        if np.array(pose_pred)[0][2][0] > int(board_end_coord[0]):
            if handstand:
                distance = math.dist(np.array(pose_pred)[0][15], board_end_coord) < math.dist(np.array(pose_pred)[0][14], np.array(pose_pred)[0][15]) * 1.5
            else:
                distance = math.dist(np.array(pose_pred)[0][5], board_end_coord) < math.dist(np.array(pose_pred)[0][5], np.array(pose_pred)[0][4]) * 1.5
            
            return distance
        # if left of board end
        else:
            return True
    else:
        # if right of board end
        if np.array(pose_pred)[0][2][0] > int(board_end_coord[0]):
            return True
        # if left of board end
        else:
            if handstand:
                distance = math.dist(np.array(pose_pred)[0][10], board_end_coord) < math.dist(np.array(pose_pred)[0][11], np.array(pose_pred)[0][10]) * 1.5
            else:
                distance = math.dist(np.array(pose_pred)[0][0], board_end_coord) < math.dist(np.array(pose_pred)[0][1], np.array(pose_pred)[0][0]) * 1.5
            return distance

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
        if angle < 70:
            three_in_a_row += 1
            if three_in_a_row >=3:
                return 'tuck'
        else:
            three_in_a_row =0
    if twist_counter_full_dive(dive_data) > 0 and som_counter_full_dive(dive_data)[0] < 5:
        return 'free'
    return 'pike'


def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    # Calculate the squared distance from point (px, py) to the line segment [(x1, y1), (x2, y2)]
    def sqr_distance_point_to_segment():
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_length_sq == 0:
            return (px - x1)**2 + (py - y1)**2
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        return ((px - (x1 + t * (x2 - x1)))**2 + (py - (y1 + t * (y2 - y1)))**2)

    # Calculate the closest point on the line segment to the given point (px, py)
    def closest_point_on_line_segment():
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_length_sq == 0:
            return x1, y1
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        return closest_x, closest_y

    closest_point = closest_point_on_line_segment()
    distance = math.sqrt(sqr_distance_point_to_segment())

    return closest_point, distance

def min_distance_from_line_to_circle(line_start, line_end, circle_center, circle_radius):
    closest_point, distance = distance_point_to_line_segment(circle_center[0], circle_center[1],
                                                             line_start[0], line_start[1],
                                                             line_end[0], line_end[1])

    min_distance = max(0, distance - circle_radius)
    return min_distance

def twister(pose_pred, prev_pose_pred=None, in_petal=False, petal_count=0, outer=10, inner=9, valid=17, middle=0.5):
    if pose_pred is None:
        return petal_count, in_petal
    min_dist = 0
    pose_pred = pose_pred[0]
    vector1 = [pose_pred[2][0] - pose_pred[3][0], 0-(pose_pred[2][1] - pose_pred[3][1])]
    if prev_pose_pred is not None:
        prev_pose_pred = prev_pose_pred[0]
        prev_pose_pred = [prev_pose_pred[2][0] - prev_pose_pred[3][0], 0-(prev_pose_pred[2][1] - prev_pose_pred[3][1])]
        min_dist = min_distance_from_line_to_circle(prev_pose_pred, vector1, (0, 0), middle)
    if np.linalg.norm(vector1) > valid:
        return petal_count, in_petal
    if min_dist is not None and in_petal and np.linalg.norm(vector1) > outer and min_dist == 0: 
        petal_count += 1
    elif not in_petal and np.linalg.norm(vector1) > outer: 
        in_petal = True
        petal_count += 1
    elif in_petal and np.linalg.norm(vector1) < inner:
        in_petal = False
    return petal_count, in_petal


def twist_counter(pose_pred, prev_pose_pred=None, in_petal=False, petal_count=0):
    valid = 17
    outer = 10
    inner = 9
    if pose_pred is None:
        return petal_count, in_petal
    min_dist = 0
    pose_pred = pose_pred[0]
    vector1 = [pose_pred[2][0] - pose_pred[3][0], 0-(pose_pred[2][1] - pose_pred[3][1])]
    if prev_pose_pred is not None:
        prev_pose_pred = prev_pose_pred[0]
        prev_pose_pred = [prev_pose_pred[2][0] - prev_pose_pred[3][0], 0-(prev_pose_pred[2][1] - prev_pose_pred[3][1])]
        min_dist = min_distance_from_line_to_circle(prev_pose_pred, vector1, (0, 0), 0.5)
    if np.linalg.norm(vector1) > valid:
        return petal_count, in_petal
    if min_dist is not None and in_petal and np.linalg.norm(vector1) > outer and min_dist == 0: 
        petal_count += 1
    elif not in_petal and np.linalg.norm(vector1) > outer: 
        in_petal = True
    elif in_petal and np.linalg.norm(vector1) < inner:
        in_petal = False
        petal_count += 1
    return petal_count, in_petal


def twist_counter_full_dive(dive_data, visualize=False):
    dist_hip = []
    prev_pose_pred = None
    in_petal=False
    petal_count=0
    scale = get_scale_factor(dive_data)
    valid = scale / 1.5
    outer = scale / 3.2
    inner = scale / 3.4
    middle = 0.5
    next_next_pose_pred = dive_data['pose_pred'][4]
    for i in range(len(dive_data['pose_pred'])):
        pose_pred = dive_data['pose_pred'][i]
        if i < len(dive_data['pose_pred']) - 1:
            next_pose_pred = dive_data['pose_pred'][i + 1]
        if i < len(dive_data['pose_pred']) - 4 and next_next_pose_pred is not None:
            next_next_pose_pred = dive_data['pose_pred'][i + 4]
        if pose_pred is None or dive_data['on_boards'][i] == 1 or dive_data['position_tightness'][i] <= 80 or next_next_pose_pred is None:
            continue
        petal_count, in_petal = twister(pose_pred, prev_pose_pred=prev_pose_pred, in_petal=in_petal, petal_count=petal_count, outer=outer, inner=inner, middle=middle, valid=valid)
        prev_pose_pred = pose_pred
        if visualize:
            pose_pred = pose_pred[0]
            dist_hip.append([pose_pred[2][0] - pose_pred[3][0], 0-(pose_pred[2][1] - pose_pred[3][1])])
    if visualize:
        dist_hip = np.array(dist_hip)
        plt.plot(dist_hip[:, 0], dist_hip[:, 1], label="right-to-left hip")
        circle1 = plt.Circle((0, 0), outer, fill=False)
        plt.gca().add_patch(circle1)
        circle2 = plt.Circle((0, 0), inner, fill=False)
        plt.gca().add_patch(circle2)
        circle3 = plt.Circle((0, 0), valid, fill=False)
        plt.gca().add_patch(circle3)
        plt.legend()
        plt.show()
    return petal_count

def rotation_direction_som(vector1, vector2, threshold=0.4):
    # Calculate the determinant to determine rotation direction
    determinant = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    mag1= np.linalg.norm(vector1)
    mag2= np.linalg.norm(vector2)
    norm_det = determinant/(mag1*mag2)
    theta = np.arcsin(norm_det)
    return math.degrees(theta)
def is_handstand(dive_data):
    first_frame_pose_pred = dive_data['pose_pred'][0]
    handstand = False
    if first_frame_pose_pred[0][6][1] < first_frame_pose_pred[0][7][1]:
        handstand = True
    return handstand


def som_counter(pose_pred=None, prev_pose_pred=None, half_som_count=0, handstand=False):
    if pose_pred is None:
        return half_som_count, True
    pose_pred = pose_pred[0]
    vector1 = [pose_pred[7][0] - pose_pred[6][0], 0-(pose_pred[7][1] - pose_pred[6][1])] # flip y axis
    if (not handstand and half_som_count % 2 == 0) or (handstand and half_som_count %2 == 1):  
        vector2 = [0, -1]
    else:
        vector2 = [0, 1]
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    current_angle = math.degrees(np.arccos(dot_product))
    if prev_pose_pred is not None:
        prev_pose_pred = prev_pose_pred[0]
        prev_vector = [prev_pose_pred[7][0] - prev_pose_pred[6][0], 0-(prev_pose_pred[7][1] - prev_pose_pred[6][1])] # flip y axis
        prev_unit_vector = prev_vector / np.linalg.norm(prev_vector)
        prev_angle_diff = math.degrees(np.arccos(np.dot(unit_vector_1, prev_unit_vector)))
        if prev_angle_diff > 115:
            return half_som_count, True
    if current_angle <= 80:
        half_som_count += 1
    return half_som_count, False

    
def som_counter_full_dive(dive_data, visualize=False):
    half_som_count = 0
    dist_body = []
    handstand = is_handstand(dive_data)
    next_next_pose_pred = dive_data['pose_pred'][2]
    prev = None
    for i in range(len(dive_data['pose_pred'])):
        pose_pred = dive_data['pose_pred'][i]
        if i < len(dive_data['pose_pred']) - 2 and next_next_pose_pred is not None:
            next_next_pose_pred = dive_data['pose_pred'][i + 2]
        if pose_pred is None or next_next_pose_pred is None or dive_data['on_boards'][i] == 1: 
            continue
        pose_pred = pose_pred[0]
        vector1 = [pose_pred[7][0] - pose_pred[6][0], 0-(pose_pred[7][1] - pose_pred[6][1])] 
        if (not handstand and half_som_count % 2 == 0) or (handstand and half_som_count % 2 == 1):  
            vector2 = [0, -1]
        else:
            vector2 = [0, 1]        
        sensitivity = 115
        if prev is not None and find_angle(vector1, prev) > sensitivity:
            continue
        is_clockwise = is_rotating_clockwise(dive_data)
        if prev is not None and ((is_clockwise and rotation_direction_som(vector1, prev)<0) or (not is_clockwise and rotation_direction_som(vector1, prev)>0)):
            continue
        angle = find_angle(vector1, vector2)
        if angle <= 75:
            half_som_count += 1
        if visualize:
            dist_body.append([pose_pred[7][0] - pose_pred[6][0], 0-(pose_pred[7][1] - pose_pred[6][1])])
        prev = vector1
    if visualize:
        dist_body = np.array(dist_body)
        plt.plot(dist_body[:, 0], dist_body[:, 1], label="pelvis-to-thorax")
        plt.xlabel("x-coord")
        plt.ylabel("y-coord")
        plt.legend()
        plt.show()
    return half_som_count, handstand

def getDiveInfo(diveNum):
    handstand = (diveNum[0] == '6')
    expected_som = int(diveNum[2])
    if len(diveNum) == 5:
        expected_twists = int(diveNum[3])
    else:
        expected_twists = 0
    if diveNum[0] == '1' or diveNum[0] == '3' or diveNum[:2] == '51' or diveNum[:2] == '53' or diveNum[:2] == '61' or diveNum[:2] == '63':
        back_facing = False
    else:
        back_facing = True
    if diveNum[0] == '1' or diveNum[:2] == '51' or diveNum[:2] == '61':
        expected_direction = 'front'
    elif diveNum[0] == '2' or diveNum[:2] == '52' or diveNum[:2] == '62':
        expected_direction = 'back'
    elif diveNum[0] == '3' or diveNum[:2] == '53' or diveNum[:2] == '63':
        expected_direction = 'reverse'
    elif diveNum[0] == '4':
        expected_direction = 'inward'
    if diveNum[-1] == 'b':
        position = 'pike'
    elif diveNum[-1] == 'c':
        position = 'tuck'
    else:
        position = 'free'
    return handstand, expected_som, expected_twists, back_facing, expected_direction, position

def get_direction(dive_data):
    clockwise = is_rotating_clockwise(dive_data)
    board_side = dive_data['board_side']
    if board_side == "right":
        back_facing = is_back_facing(dive_data, 'right')
        if back_facing and clockwise:
            direction = 'inward'
        elif back_facing and not clockwise:
            direction = 'back'
        elif not back_facing and clockwise:
            direction = 'reverse'
        elif not back_facing and not clockwise:
            direction = 'front'
    else:
        back_facing = is_back_facing(dive_data, 'left')
        if back_facing and clockwise:
            direction = 'back'
        elif back_facing and not clockwise:
            direction = 'inward'
        elif not back_facing and clockwise:
            direction = 'front'
        elif not back_facing and not clockwise:
            direction = 'reverse'
    return direction

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
