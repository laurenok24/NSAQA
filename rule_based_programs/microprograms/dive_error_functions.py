"""
dive_error_functions.py
Author: Lauren Okamoto

AQA functions.
"""

import torch
import numpy as np
import math
import cv2
import sys, os
from matplotlib import image
from matplotlib import pyplot as plt
from math import atan
from models.pose_estimator.pose_estimator_model_setup import get_pose_estimation
from models.detectron2.detectors import get_platform_detector, get_splash_detector
from detectron2.utils.visualizer import Visualizer

################## HELPER FUNCTIONS ##################
def slope(x1, y1, x2, y2):
    if x1 == x2:
        return "undefined"
    return (y2-y1)/(x2-x1)

# Function to find the angle between two lines
def findAngle(M1, M2):
    vertical_line = False
    if M1 == "undefined":
        M1 = 0
        vertical_line = True
    if M2 == "undefined":
        M2 = 0
        vertical_line = True
    PI = 3.14159265
    angle = abs((M2 - M1) / (1 + M1 * M2))
    ret = atan(angle)
    val = (ret * 180) / PI
    if vertical_line:
        return 90 - round(val,4)
    return (round(val, 4))

def find_which_side_board_on(output):
    pred_classes = output['instances'].pred_classes.cpu().numpy()
    platforms = np.where(pred_classes == 0)[0]
    scores = output['instances'].scores[platforms]
    if len(scores) == 0:
      return
    pred_masks = output['instances'].pred_masks[platforms]
    max_instance = torch.argmax(scores)
    pred_mask = np.array(pred_masks[max_instance].cpu()) 
    for i in range(len(pred_mask[0])//2):
        if sum(pred_mask[:, i]) != 0:
            return "left"
        elif sum(pred_mask[:, len(pred_mask[0]) - i - 1]) != 0:
            return "right"
    return None

def board_end(output, board_side=None):
    pred_classes = output['instances'].pred_classes.cpu().numpy()
    platforms = np.where(pred_classes == 0)[0]
    scores = output['instances'].scores[platforms]
    if len(scores) == 0:
      return
    pred_masks = output['instances'].pred_masks[platforms]
    max_instance = torch.argmax(scores)
    pred_mask = np.array(pred_masks[max_instance].cpu()) # splash instance with highest confidence
    # need to figure out whether springboard is on left or right side of frame, then need to find where the edge is
    if board_side is None:
        board_side = find_which_side_board_on(output)
    if board_side == "left":
      for i in range(len(pred_mask[0]) - 1, -1, -1):
        if sum(pred_mask[:, i]) != 0:
          trues = np.where(pred_mask[:, i])[0]
          return (i, min(trues))
    if board_side == "right":
      for i in range(len(pred_mask[0])):
        if sum(pred_mask[:, i]) != 0:
          trues = np.where(pred_mask[:, i])[0]
          return (i, min(trues))
    return None

# Splash helper function
def get_splash_pred_mask(output):
  pred_classes = output['instances'].pred_classes.cpu().numpy()
  splashes = np.where(pred_classes == 0)[0]
  scores = output['instances'].scores[splashes]
  if len(scores) == 0:
    return None
  pred_masks = output['instances'].pred_masks[splashes]
  max_instance = torch.argmax(scores)
  pred_mask = np.array(pred_masks[max_instance].cpu()) 
  return pred_mask

# Splash helper function that finds the splash instance with the highest percent confidence
# and returns the
def splash_area_percentage(output, pred_mask=None):
  if pred_mask is None:
    return
  # loops over pixels to get sum of splash pixels
  totalSum = 0
  for j in range(len(pred_mask)):
    totalSum += pred_mask[j].sum()
  # return percentage of image that is splash
  return totalSum/(len(pred_mask) * len(pred_mask[0]))

def draw_two_coord(im, coord1, coord2, filename):
    print("hello, im in the drawing func")
    image = cv2.circle(im, (int(coord1[0]),int(coord1[1])), radius=5, color=(0, 0, 255), thickness=-1)
    image = cv2.circle(image, (int(coord2[0]),int(coord2[1])), radius=5, color=(0, 255, 0), thickness=-1)
    print(filename)
    if not cv2.imwrite(filename, image):
        print(filename)
        print("file failed to write")

def draw_board_end_coord(im, coord):
    print("hello, im in the drawing func")
    image = cv2.circle(im, (int(coord[0]),int(coord[1])), radius=10, color=(0, 0, 255), thickness=-1)
    filename = os.path.join("./output/board_end/", d["file_name"][3:])
    print(filename)
    if not cv2.imwrite(filename, image):
        print(filename)
        print("file failed to write")



#######################################################

################## DIVE ERROR MICROPROGRAMS ##################

## Feet apart error ##
def applyFeetApartError(filepath, pose_pred=None, diver_detector=None, pose_model=None):
    if pose_pred is None and filepath != "":
        diver_box, pose_pred = get_pose_estimation(filepath, diver_detector=diver_detector, pose_model=pose_model)
    if pose_pred is not None:
        pose_pred = np.array(pose_pred)[0]
        average_knee = [np.mean((pose_pred[4][0], pose_pred[1][0])), np.mean((pose_pred[4][1], pose_pred[1][1]))]
        vector1 = [pose_pred[5][0] - average_knee[0], pose_pred[5][1] - average_knee[1]]
        vector2 = [pose_pred[0][0] - average_knee[0], pose_pred[0][1] - average_knee[1]]
        unit_vector_1 = vector1 / np.linalg.norm(vector1)
        unit_vector_2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = math.degrees(np.arccos(dot_product))
        return angle
    else:
        return None

## Calculates hip bend for somersault tightness & twist straightness errors ##
def applyPositionTightnessError(filepath, pose_pred=None, diver_detector=None, pose_model=None):
    if pose_pred is None and filepath != "":
        diver_box, pose_pred = get_pose_estimation(filepath, diver_detector=diver_detector, pose_model=pose_model)
    if pose_pred is not None:
        pose_pred = np.array(pose_pred)[0]
        vector1 = [pose_pred[7][0] - pose_pred[2][0], pose_pred[7][1] - pose_pred[2][1]]
        vector2 = [pose_pred[1][0] - pose_pred[2][0], pose_pred[1][1] - pose_pred[2][1]]
        unit_vector_1 = vector1 / np.linalg.norm(vector1)
        unit_vector_2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = math.degrees(np.arccos(dot_product))
        return angle
    else:
        return None

## Distance from board error ##
def calculate_distance_from_platform_for_one_frame(filepath, im=None, visualize=False, dive_folder_num="", platform_detector=None, pose_pred=None, diver_detector=None, pose_model=None, board_end_coord=None, board_side=None):
    if platform_detector is None:
        platform_detector = get_platform_detector()
    if pose_pred is None:
        diver_box, pose_pred = get_pose_estimation(filepath, image_bgr=im, diver_detector=diver_detector, pose_model=pose_model)
    if im is None and filepath != "":
        im = cv2.imread(filepath)
    if board_end_coord is None:
        outputs = platform_detector(im)
        board_end_coord = board_end(outputs, board_side=board_side)
    minDist = None
    if board_end_coord is not None and pose_pred is not None and len(board_end_coord) == 2:
        minDist = float('inf')
        for i in range(len(np.array(pose_pred)[0])):
            distance = math.dist(np.array(pose_pred)[0][i], np.array(board_end_coord))
            if distance < minDist:
                minDist = distance
                minJoint = i
        if visualize:
            file_name = filepath.split('/')[-1]
            folder = "./output/data/distance_from_board/{}".format(dive_folder_num)
            out_filename = os.path.join(folder, file_name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            draw_two_coord(im, board_end_coord, np.array(pose_pred)[0][minJoint], filename=out_filename)
    return minDist

## Over-rotation error ##
def over_rotation(filepath, pose_pred=None, diver_detector=None, pose_model=None):
    if pose_pred is None and filepath != "":
        diver_box, pose_pred = get_pose_estimation(filepath, diver_detector=diver_detector, pose_model=pose_model)
    if pose_pred is not None:
        pose_pred = np.array(pose_pred)[0]
        vector1 = [(pose_pred[0][0] - pose_pred[2][0]), 0-(pose_pred[0][1] - pose_pred[2][1])]        
        vector2 = [-1, 0]
        unit_vector_1 = vector1 / np.linalg.norm(vector1)
        unit_vector_2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = math.degrees(np.arccos(dot_product))
        return angle
    else:
        return None

## Splash size error ##
def get_splash_from_one_frame(filepath, im=None, predictor=None, visualize=False, dive_folder_num=""):
    if predictor is None:
      predictor=get_splash_detector()
    if im is None:
      im = cv2.imread(filepath)
    outputs = predictor(im)
    pred_mask = get_splash_pred_mask(outputs)
    area = splash_area_percentage(outputs, pred_mask=pred_mask)
    if area is None:
        return None, None
    if visualize:
        pred_boxes = outputs['instances'].pred_boxes
        print("pred_boxes", pred_boxes)
        for box in pred_boxes:
          image = cv2.rectangle(im, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), color=(0, 0, 255), thickness=2)
          out_folder= "./output/data/splash/{}".format(dive_folder_num)
          if not os.path.exists(out_folder):
              os.makedirs(out_folder)
          filename = os.path.join(out_folder, filepath.split('/')[-1])
          if not cv2.imwrite(filename, image):
              print('no image written to', filename)
          break
        
    return area.tolist(), pred_mask
