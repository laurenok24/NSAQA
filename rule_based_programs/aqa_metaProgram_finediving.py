"""
aqa_metaProgram_finediving.py
Author: Lauren Okamoto
"""

from rule_based_programs.microprograms.dive_error_functions import *
from rule_based_programs.microprograms.temporal_segmentation_functions import *
from rule_based_programs.microprograms.dive_recognition_functions import *
from rule_based_programs.scoring_functions import get_scale_factor
from models.detectron2.detectors import get_platform_detector, get_diver_detector, get_splash_detector
from models.pose_estimator.pose_estimator_model_setup import get_pose_estimation, get_pose_model
import pickle
import os, math
import numpy as np
import cv2
import argparse

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

def aqa_metaprogram_finediving(first_folder, second_folder, diveNum, board_side=None, platform_detector=None, splash_detector=None, diver_detector=None, pose_model=None):
    handstand, expected_som, expected_twists, back_facing, expected_direction, position = getDiveInfo_from_diveNum(diveNum)
    dive_data = {}
    takeoff = []
    twist = []
    som = []
    entry = []
    distance_from_board = []
    position_tightness = []
    feet_apart = []
    over_under_rotation = []
    splash = []
    pose_preds = []
    diver_boxes = []
    above_boards = []
    on_boards = []
    som_counts = []
    twist_counts = []
    board_end_coords = []
    plat_outputs = []
    board_sides = []
    splash_pred_masks = []
    above_board = True
    on_board = True
    if platform_detector is None:
        platform_detector = get_platform_detector()
    if splash_detector is None:
        splash_detector = get_splash_detector()
    if diver_detector is None:
        diver_detector = get_diver_detector()
    if pose_model is None:
        pose_model = get_pose_model()
    key = (first_folder, int(second_folder))
    dive_folder_num = "{}_{}".format(first_folder, second_folder)
    directory = './FineDiving/datasets/FINADiving_MTL_256s/{}/{}/'.format(first_folder, second_folder)
    file_names = os.listdir(directory)

    ## find board_side
    if board_side is None:
        for i in range(len(file_names)):
            filepath = directory + file_names[i]
            if file_names[i][-4:] != ".jpg":
                continue
            im = cv2.imread(filepath)
            plat_output = platform_detector(im)
            board_side = find_which_side_board_on(plat_output)
            if board_side is not None:
                board_sides.append(board_side)
        dive_data['board_sides'] = board_sides
        board_sides.sort()
        board_side = board_sides[len(board_sides)//2]
    dive_data['board_side'] = board_side

    prev_pred = None
    som_prev_pred = None
    half_som_count=0
    petal_count = 0
    in_petal = False
    for i in range(len(file_names)):
        filepath = directory + file_names[i]
        if file_names[i][-4:] != ".jpg":
            continue
        diver_box, pose_pred = get_pose_estimation(filepath, diver_detector=diver_detector, pose_model=pose_model)
        diver_boxes.append(diver_box)
        pose_preds.append(pose_pred)

        calculated_half_som_count, skip = som_counter(pose_pred, prev_pose_pred=som_prev_pred, half_som_count=half_som_count, handstand=handstand)
        if not skip:
            som_prev_pred = pose_pred
        calculated_petal_count, calculated_in_petal = twist_counter(pose_pred, prev_pose_pred=prev_pred, in_petal=in_petal, petal_count=petal_count)
        im = cv2.imread(filepath)
        plat_output = platform_detector(im)
        plat_outputs.append(plat_output)
        board_end_coord = board_end(plat_output, board_side=board_side)
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
        calculated_takeoff = takeoff_microprogram_one_frame(filepath, above_board=above_board, on_board=on_board, pose_pred=pose_pred)
        calculated_twist = twist_microprogram_one_frame(filepath, on_board=on_board, pose_pred=pose_pred, expected_twists=expected_twists, petal_count=petal_count, expected_som=expected_som, half_som_count=half_som_count, diver_detector=diver_detector, pose_model=pose_model)
        calculated_som = somersault_microprogram_one_frame(filepath, pose_pred=pose_pred, on_board=on_board, expected_som=expected_som, half_som_count=half_som_count, expected_twists=expected_twists, petal_count=petal_count, diver_detector=diver_detector, pose_model=pose_model)
        calculated_entry = entry_microprogram_one_frame(filepath, above_board=above_board, on_board=on_board, pose_pred=pose_pred, expected_twists=expected_twists, petal_count=petal_count, expected_som=expected_som, half_som_count=half_som_count, splash_detector=splash_detector, visualize=False, dive_folder_num=dive_folder_num)
        if calculated_som == 1:
            half_som_count = calculated_half_som_count
        elif calculated_twist == 1:
            half_som_count = calculated_half_som_count
            petal_count = calculated_petal_count
            in_petal = calculated_in_petal
        dist = calculate_distance_from_platform_for_one_frame(filepath, visualize=False, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model, board_end_coord=board_end_coord, platform_detector=platform_detector) # saves photo to ./output/data/distance_from_board/
        distance_from_board.append(dist)
        position_tightness.append(applyPositionTightnessError(filepath, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        feet_apart.append(applyFeetApartError(filepath, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        over_under_rotation.append(over_rotation(filepath, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model))
        splash_area, splash_pred_mask = get_splash_from_one_frame(filepath, predictor=splash_detector, visualize=False)
        splash.append(splash_area)
        splash_pred_masks.append(splash_pred_mask)    
        takeoff.append(calculated_takeoff)
        twist.append(calculated_twist)
        som.append(calculated_som)
        entry.append(calculated_entry)
        som_counts.append(half_som_count)
        twist_counts.append(petal_count)
        prev_pred = pose_pred

    dive_data['pose_pred'] = pose_preds
    dive_data['takeoff'] = takeoff
    dive_data['twist'] = twist
    dive_data['som'] = som
    dive_data['entry'] = entry
    dive_data['distance_from_board'] = distance_from_board
    dive_data['position_tightness'] = position_tightness
    dive_data['feet_apart'] = feet_apart
    dive_data['over_under_rotation'] = over_under_rotation
    dive_data['splash'] = splash
    dive_data['above_boards'] = above_boards
    dive_data['on_boards'] = on_boards
    dive_data['som_counts'] = som_counts
    dive_data['twist_counts'] = twist_counts
    dive_data['board_end_coords'] = board_end_coords
    dive_data['diver_boxes'] = diver_boxes
    dive_data['splash_pred_masks'] = splash_pred_masks
    dive_data['board_side'] = board_side
    dive_data['is_handstand'] = handstand
    dive_data['direction'] = expected_direction
    dive_data['plat_outputs'] = plat_outputs
    return dive_data

if __name__ == '__main__':
    # Set up command-line arguments
    new_parser = argparse.ArgumentParser(description="Extract dive data to be used for scoring.")
    new_parser.add_argument("FineDiving_key", type=str, nargs=2, help="key from FineDiving Dataset (e.g. 01 1)")
    meta_program_args = new_parser.parse_args()

    # Fine-grained annotations from FineDiving Dataset
    with open('FineDiving/Annotations/fine-grained_annotation_aqa.pkl', 'rb') as f:
        dive_annotation_data = pickle.load(f)

    key = tuple(meta_program_args.FineDiving_key)
    key = (key[0], int(key[1]))
    print(key)
    platform_detector = get_platform_detector()
    splash_detector = get_splash_detector()
    diver_detector = get_diver_detector()
    pose_model = get_pose_model()
    diveNum = dive_annotation_data[key][0]
    print(diveNum)
    dive_data = aqa_metaprogram_finediving(key[0], key[1], diveNum, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)

    save_path = "./output/{}_{}.pkl".format(key[0], key[1])
    with open(save_path, 'wb') as f:
        print("saving data into " + save_path)
        pickle.dump(dive_data, f)
