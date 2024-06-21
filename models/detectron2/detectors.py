"""
detectors.py
Author: Lauren Okamoto

Code used to initialize the object detector models to be used for inference.
"""

import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))

import detectron2
import cv2

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances


def get_diver_detector():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "./models/detectron2/model_weights/"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "diver_model_final.pth") 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    diver_detector = DefaultPredictor(cfg)
    return diver_detector


def get_platform_detector():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./models/detectron2/model_weights/"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "plat_model_final.pth") 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    platform_detector = DefaultPredictor(cfg)
    return platform_detector

def get_splash_detector():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./models/detectron2/model_weights/"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "splash_model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    splash_detector = DefaultPredictor(cfg)
    return splash_detector