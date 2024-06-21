"""
temporal_segmentation_functions.py
Author: Lauren Okamoto

Temporal Segmentation Microprograms for detecting start/takeoff, somersault, twist, and entry phases
"""

from rule_based_programs.microprograms.dive_error_functions import get_splash_from_one_frame, applyPositionTightnessError
import numpy as np

"""
    Start/Takeoff Phase Microprogram

    Parameters:
    - filepath (str): file path where the frame is located
    - above_board (bool): True if the diver is above the board at this frame
    - on_board (bool): True if the diver is on the board at this frame
    - pose_pred: pose estimation of the diver at this frame (None if no diver detected)

    Returns:
    - 0 if frame is not in start/takeoff phase
    - 1 if frame is in start/takeoff phase
"""
def takeoff_microprogram_one_frame(filepath, above_board, on_board, pose_pred=None):
    if not above_board:
        return 0
    if on_board:
        return 1
    return 0

"""
    Somersault Phase Microprogram

    Parameters:
    - filepath (str): file path where the frame is located
    - on_board (bool): True if the diver is on the board at this frame
    - expected_som (int): number of somersaults in full dive (from action recognition)
    - half_som_count (int): number of somersaults counted by this frame
    - expected_twists (int): number of twists in full dive (from action recognition)
    - petal_count (int): number of twists counted by this frame
    - pose_pred: pose estimation of the diver at this frame (None if no diver detected)
    - diver_detector: diver detector model
    - pose_model: pose estimator model

    Returns:
    - 0 if frame is not in somersault phase
    - 1 if frame is in somersault phase
"""
def somersault_microprogram_one_frame(filepath, on_board, expected_som, half_som_count, expected_twists, petal_count, pose_pred=None, diver_detector=None, pose_model=None):
    if on_board:
        return 0
    if expected_som <= half_som_count:
        return 0
    # if not done with som or twists, need to determine if som or twist
    angle = applyPositionTightnessError(filepath, pose_pred, diver_detector=diver_detector, pose_model=pose_model)
    if angle is None:
        return 0
    # if not done with som but done with twists
    if expected_som > half_som_count and expected_twists <= petal_count:
        return 1
    # print("angle:", angle)
    if angle <= 80:
        return 1
    else:
        return 0

"""
    Twist Phase Microprogram

    Parameters:
    - filepath (str): file path where the frame is located
    - on_board (bool): True if the diver is on the board at this frame
    - expected_twists (int): number of twists in full dive (from action recognition)
    - petal_count (int): number of twists counted by this frame
    - expected_som (int): number of somersaults in full dive (from action recognition)
    - half_som_count (int): number of somersaults counted by this frame
    - pose_pred: pose estimation of the diver at this frame (None if no diver detected)
    - diver_detector: diver detector model
    - pose_model: pose estimator model

    Returns:
    - 0 if frame is not in twist phase
    - 1 if frame is in twist phase
"""
def twist_microprogram_one_frame(filepath, on_board, expected_twists, petal_count, expected_som, half_som_count, pose_pred=None, diver_detector=None, pose_model=None):
    if on_board:
        return 0
    if expected_twists <= petal_count or expected_som <= half_som_count:
        return 0
    angle = applyPositionTightnessError(filepath, pose_pred=pose_pred, diver_detector=diver_detector, pose_model=pose_model)
    if angle is None:
        return 0
    if angle > 80:
        return 1
    else:
        return 0

"""
    Entry Phase Microprogram

    Parameters:
    - filepath (str): file path where the frame is located
    - above_board (bool): True if the diver is above the board at this frame
    - on_board (bool): True if the diver is on the board at this frame
    - pose_pred: pose estimation of the diver at this frame (None if no diver detected)
    - expected_twists (int): number of twists in full dive (from action recognition)
    - petal_count (int): number of twists counted by this frame
    - expected_som (int): number of somersaults in full dive (from action recognition)
    - half_som_count (int): number of somersaults counted by this frame
    - frame: Pil Image of frame
    - splash_detector: splash detector model
    - visualize: True if you want to save the splash segmentation mask prediction to an image
    - dive_folder_num: if visualize is true, this is where the image will be saved

    Returns:
    - 0 if frame is not in entry phase
    - 1 if frame is in entry phase
"""
def entry_microprogram_one_frame(filepath, above_board, on_board, pose_pred, expected_twists, petal_count, expected_som, half_som_count, frame=None, splash_detector=None,  visualize=False, dive_folder_num=None):
    if above_board:
        return 0
    if on_board:
        return 0
    splash = get_splash_from_one_frame(filepath, im=frame, predictor=splash_detector, visualize=visualize, dive_folder_num=dive_folder_num)
    if splash:
        return 1
    # if completed with somersaults, we know we're in entry phase
    if not expected_som > half_som_count:
        return 1
    if expected_twists > petal_count or expected_som > half_som_count:
        return 0
    return 1
