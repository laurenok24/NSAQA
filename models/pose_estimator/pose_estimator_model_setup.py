"""
pose_estimator_model_setup.py
Author: Lauren Okamoto

Code used to initialize the pose estimator model, HRNet, combined with the diver detectron2 model to be used for diver pose inference.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil
import sys

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
sys.path.append('./deep-high-resolution-net.pytorch/lib')
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

import distutils.core

from models.detectron2.diver_detector_setup import get_diver_detector
from models.pose_estimator.pose_hrnet import get_pose_net


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    bottom_left_corner = (box[0].data.cpu().item(), box[1].data.cpu().item())
    top_right_corner = (box[2].data.cpu().item(), box[3].data.cpu().item())
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5
    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', type=str, default='./models/pose_estimator/cfg/w32_256x256_adam_lr1e-3.yaml')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.opts = ''
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    model_input = cv2.warpAffine(
    image,
    trans,
    (256, 256),
    flags=cv2.INTER_LINEAR)

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))
        return preds

def get_pose_model():
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    args = parse_args()
    update_config(cfg, args)
    pose_model = get_pose_net(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()
    return pose_model

def get_pose_estimation(filepath, image_bgr=None, diver_detector=None, pose_model=None):
    if image_bgr is None:
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            print("ERROR: image {} does not exist".format(filepath))
            return None
    if diver_detector is None:
        diver_detector = get_diver_detector()
    
    if pose_model is None:
        pose_model = get_pose_model()

    image = image_bgr[:, :, [2, 1, 0]]

    outputs = diver_detector(image_bgr)
    scores = outputs['instances'].scores
    pred_boxes = []
    if len(scores) > 0:
        pred_boxes = outputs['instances'].pred_boxes           
    
    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            box = box.detach().cpu().numpy()
            return box, get_pose_estimation_prediction(pose_model, image_pose, center, scale)
    return None, None
