"""
aqa_metaProgram.py
Author: Lauren Okamoto
"""

from rule_based_programs.microprograms.dive_error_functions import *
from rule_based_programs.microprograms.temporal_segmentation_functions import *
from rule_based_programs.microprograms.dive_recognition_functions import *
from rule_based_programs.scoring_functions import get_scale_factor
from models.detectron2.detectors import get_platform_detector, get_diver_detector, get_splash_detector
from models.pose_estimator.pose_estimator_model_setup import get_pose_estimation, get_pose_model
import gradio as gr
import pickle
import os, sys, math
import numpy as np
import cv2
import argparse

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        exit()
    frame_skip = 1
    # a variable to keep track of the frame to be saved
    frame_count = 0
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if i > frame_skip - 1:
            frame_count += 1
            frame = cv2.resize(frame, (455, 256)) # resize takes argument (width, height)
            frames.append(frame)
            i = 0
            continue
        i += 1
    cap.release()
    return frames

def getDiveInfo_from_diveNum(diveNum):
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

def getDiveInfo_from_symbols(frames, dive_data=None, platform_detector=None, splash_detector=None, diver_detector=None, pose_model=None):
    print("Getting dive info from symbols...")
    if dive_data is None:
        print("somethings not getting passed in properly")
        dive_data = abstractSymbols(frames, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)

    # get above_boards, on_boards, and position_tightness
    above_board = True
    on_board = True
    above_boards = []
    on_boards = []
    position_tightness = []
    distances = []
    prev_board_coord = None
    for i in range(len(dive_data['pose_pred'])):
        pose_pred = dive_data['pose_pred'][i]
        board_end_coord =  dive_data['board_end_coords'][i]
        if board_end_coord is not None and prev_board_coord is not None:
            distances.append(math.dist(board_end_coord, prev_board_coord))
            if math.dist(board_end_coord, prev_board_coord) > 150:
                position_tightness.append(applyPositionTightnessError(filepath="", pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
                if above_board:
                    above_boards.append(1)
                else:
                    above_boards.append(0)
                if on_board:
                    on_boards.append(1)
                else:
                    on_boards.append(0)
                continue
        if above_board and not on_board and board_end_coord is not None and pose_pred is not None and np.array(pose_pred)[0][2][1] > int(board_end_coord[1]):
            above_board=False
        if on_board:
            handstand = is_handstand(dive_data)
            calculate_on_board = detect_on_board(board_end_coord, dive_data['board_side'], pose_pred, handstand)
            if calculate_on_board is not None and not calculate_on_board:
                on_board = False
        if above_board:
            above_boards.append(1)
        else:
            above_boards.append(0)
        if on_board:
            on_boards.append(1)
        else:
            on_boards.append(0)
        prev_board_coord = board_end_coord
        position_tightness.append(applyPositionTightnessError(filepath="", pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
    dive_data['on_boards'] = on_boards
    dive_data['above_boards'] = above_boards
    dive_data['position_tightness'] = position_tightness

    ## handstand and som_count##
    expected_som, handstand = som_counter_full_dive(dive_data)

    ## twist_count
    expected_twists = twist_counter_full_dive(dive_data)

    ## direction: front, back, reverse, inward
    expected_direction = get_direction(dive_data)

    return handstand, expected_som, expected_twists, expected_direction, dive_data


def abstractSymbols(frames, progress=gr.Progress(), platform_detector=None, splash_detector=None, diver_detector=None, pose_model=None):
    print("Abstracting symbols...")
    splashes = []
    pose_preds = []
    board_sides = []
    plat_outputs = []
    diver_boxes = []
    splash_pred_masks = []
    if platform_detector is None:
        platform_detector = get_platform_detector()
    if splash_detector is None:
        splash_detector = get_splash_detector()
    if diver_detector is None:
        diver_detector = get_diver_detector()
    if pose_model is None:
        pose_model = get_pose_model()
    num_frames = len(frames)
    i = 0
    for frame in frames:
        progress(i/num_frames, desc="Abstracting Symbols")
        plat_output = platform_detector(frame)
        plat_outputs.append(plat_output)
        board_side = find_which_side_board_on(plat_output)
        if board_side is not None:
            board_sides.append(board_side)
        diver_box, pose_pred = get_pose_estimation(filepath="", image_bgr=frame, diver_detector=diver_detector, pose_model=pose_model)
        pose_preds.append(pose_pred)
        diver_boxes.append(diver_box)
        splash_area, splash_pred_mask = get_splash_from_one_frame(filepath="", im=frame, predictor=splash_detector, visualize=False)
        splash_pred_masks.append(splash_pred_mask)
        splashes.append(splash_area)
        i+=1
    dive_data = {}
    dive_data['plat_outputs'] = plat_outputs
    dive_data['pose_pred'] = pose_preds
    dive_data['splash'] = splashes
    dive_data['splash_pred_masks'] = splash_pred_masks
    dive_data['board_sides'] = board_sides
    board_sides.sort()
    board_side = board_sides[len(board_sides)//2]
    dive_data['board_side'] = board_side
    dive_data['diver_boxes'] = diver_boxes

    # get board_end_coords
    board_end_coords = []
    for plat_output in dive_data['plat_outputs']:
        board_end_coord = board_end(plat_output, board_side=dive_data['board_side'])
        board_end_coords.append(board_end_coord)
    dive_data['board_end_coords'] = board_end_coords

    return dive_data

def aqa_metaprogram(frames, dive_data, progress=gr.Progress(), diveNum="", board_side=None, platform_detector=None, splash_detector=None, diver_detector=None, pose_model=None):
    print("AQA Metaprogram...")
    if len(frames) != len(dive_data['pose_pred']):
        raise gr.Error("Abstract Symbols first!")
    if diveNum != "":
        dive_num_given = True
        handstand, expected_som, expected_twists, back_facing, expected_direction, position = getDiveInfo_from_diveNum(diveNum)
    else:
        dive_num_given = False
        handstand, expected_som, expected_twists, expected_direction, dive_data = getDiveInfo_from_symbols(frames, dive_data=dive_data, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)

    if not dive_num_given:
        above_boards = dive_data['above_boards']
        on_boards = dive_data['on_boards']
        position_tightness = dive_data['position_tightness']
        board_end_coords = dive_data['board_end_coords']
    else:
        above_board = True
        on_board = True
        above_boards = []
        on_boards = []
        board_end_coords = []
        position_tightness = []
    splash = dive_data['splash'] 
    diver_boxes = dive_data['diver_boxes']   
    board_side = dive_data['board_side']
    pose_preds = dive_data['pose_pred']
    takeoff = []
    twist = []
    som = []
    entry = []
    distance_from_board = []
    feet_apart = []
    over_under_rotation = []
    som_counts = []
    twist_counts = []
    
    if platform_detector is None:
        platform_detector = get_platform_detector()
    if splash_detector is None:
        splash_detector = get_splash_detector()
    if diver_detector is None:
        diver_detector = get_diver_detector()
    if pose_model is None:
        pose_model = get_pose_model()

    prev_pred = None
    som_prev_pred = None
    half_som_count=0
    petal_count = 0
    in_petal = False
    num_frames = len(frames)
    for i in range(num_frames):
        progress(i/num_frames, desc="Calculating Dive Errors")
        pose_pred = pose_preds[i]
        calculated_half_som_count, skip = som_counter(pose_pred, prev_pose_pred=som_prev_pred, half_som_count=half_som_count, handstand=handstand)
        if not skip:
            som_prev_pred = pose_pred
        calculated_petal_count, calculated_in_petal = twist_counter(pose_pred, prev_pose_pred=prev_pred, in_petal=in_petal, petal_count=petal_count)
        if dive_num_given:
            outputs = platform_detector(frames[i])
            board_end_coord = board_end(outputs, board_side=board_side)
            board_end_coords.append(board_end_coord)
            if above_board and not on_board and board_end_coord is not None and pose_pred is not None and np.array(pose_pred)[0][2][1] > int(board_end_coord[1]):
                above_board=False
            if on_board and detect_on_board(board_end_coord, board_side, pose_pred, handstand) is not None and not detect_on_board(board_end_coord, board_side, pose_pred, handstand):
                on_board = False
            if above_board:
                above_boards.append(1)
            else:
                above_boards.append(0)
            if on_board:
                on_boards.append(1)
            else:
                on_boards.append(0)
        else:
            board_end_coord = board_end_coords[i]
            above_board = (above_boards[i] == 1)
            on_board = (on_boards[i] == 1)
        calculated_takeoff = takeoff_microprogram_one_frame(filepath="", above_board=above_board, on_board=on_board, pose_pred=pose_pred)
        calculated_twist = twist_microprogram_one_frame(filepath="", on_board=on_board, pose_pred=pose_pred, expected_twists=expected_twists, petal_count=petal_count, expected_som=expected_som, half_som_count=half_som_count, diver_detector=diver_detector, pose_model=pose_model)
        calculated_som = somersault_microprogram_one_frame(filepath="", pose_pred=pose_pred, on_board=on_board, expected_som=expected_som, half_som_count=half_som_count, expected_twists=expected_twists, petal_count=petal_count, diver_detector=diver_detector, pose_model=pose_model)
        calculated_entry = entry_microprogram_one_frame(filepath="", frame=frames[i], above_board=above_board, on_board=on_board, pose_pred=pose_pred, expected_twists=expected_twists, petal_count=petal_count, expected_som=expected_som, half_som_count=half_som_count, splash_detector=splash_detector, visualize=False)
        if calculated_som == 1:
            half_som_count = calculated_half_som_count
        elif calculated_twist == 1:
            half_som_count = calculated_half_som_count
            petal_count = calculated_petal_count
            in_petal = calculated_in_petal
        # distance from board
        dist = calculate_distance_from_platform_for_one_frame(filepath="", im=frames[i], visualize=False, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model, board_end_coord=board_end_coord, platform_detector=platform_detector) # saves photo to ./output/data/distance_from_board/
        distance_from_board.append(dist)
        if dive_num_given:
            position_tightness.append(applyPositionTightnessError(filepath="", pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        feet_apart.append(applyFeetApartError(filepath="", pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        over_under_rotation.append(over_rotation(filepath="", pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        takeoff.append(calculated_takeoff)
        twist.append(calculated_twist)
        som.append(calculated_som)
        entry.append(calculated_entry)
        som_counts.append(half_som_count)
        twist_counts.append(petal_count)
        prev_pred = pose_pred

    dive_data['takeoff'] = takeoff
    dive_data['twist'] = twist
    dive_data['som'] = som
    dive_data['entry'] = entry
    dive_data['distance_from_board'] = distance_from_board
    dive_data['position_tightness'] = position_tightness
    dive_data['feet_apart'] = feet_apart
    dive_data['over_under_rotation'] = over_under_rotation
    dive_data['above_boards'] = above_boards
    dive_data['on_boards'] = on_boards
    dive_data['som_counts'] = som_counts
    dive_data['twist_counts'] = twist_counts
    dive_data['board_end_coords'] = board_end_coords
    dive_data['is_handstand'] = handstand
    dive_data['direction'] = expected_direction
    return dive_data

if __name__ == '__main__':
    # Set up command-line arguments
    new_parser = argparse.ArgumentParser(description="Extract dive data to be used for scoring.")
    new_parser.add_argument("video_path", type=str, help="Path to dive video (mp4 format).")
    meta_program_args = new_parser.parse_args()

    video_path = meta_program_args.video_path
    frames = extract_frames(video_path)
    platform_detector = get_platform_detector()
    splash_detector = get_splash_detector()
    diver_detector = get_diver_detector()
    pose_model = get_pose_model()

    dive_data = abstractSymbols(frames, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)
    dive_data = aqa_metaprogram(frames, dive_data, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)

    save_path = "./output/{}.pkl".format("".join(video_path.split('.')[:-1]))
    with open(save_path, 'wb') as f:
        print("saving data into " + save_path)
        pickle.dump(dive_data, f)
