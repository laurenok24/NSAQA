"""
nsaqa_finediving.py
Author: Lauren Okamoto
"""

import pickle
from models.detectron2.detectors import get_platform_detector, get_diver_detector, get_splash_detector
from models.pose_estimator.pose_estimator_model_setup import get_pose_estimation, get_pose_model
from rule_based_programs.scoring_functions import *
from score_report_generation.generate_report_functions import *
from rule_based_programs.aqa_metaProgram_finediving import aqa_metaprogram_finediving
import argparse

def main(key):
    platform_detector = get_platform_detector()
    splash_detector = get_splash_detector()
    diver_detector = get_diver_detector()
    pose_model = get_pose_model()

    # Fine-grained annotations from FineDiving Dataset
    with open('FineDiving/Annotations/fine-grained_annotation_aqa.pkl', 'rb') as f:
        dive_annotation_data = pickle.load(f)
    diveNum = dive_annotation_data[key][0]    
    template_path = 'report_template_tables.html'
    dive_data = {}

    dive_data = aqa_metaprogram_finediving(key[0], key[1], diveNum, platform_detector=platform_detector, splash_detector=splash_detector, diver_detector=diver_detector, pose_model=pose_model)
    intermediate_scores = get_all_report_scores(dive_data)

    local_directory = "FineDiving/datasets/FINADiving_MTL_256s/{}/{}/".format(key[0], key[1]) 
    html = generate_report(template_path, intermediate_scores, local_directory)
    save_path = "./output/{}_{}_report.pkl".format(key[0], key[1])
    with open(save_path, 'w') as f:
        print("saving html report into " + save_path)
        f.write(html)


if __name__ == '__main__':
    # Set up command-line arguments
    new_parser = argparse.ArgumentParser(description="Extract dive data to be used for scoring.")
    new_parser.add_argument("FineDiving_key", type=str, nargs=2, help="key from FineDiving Dataset (e.g. 01 1)")
    meta_program_args = new_parser.parse_args()
    key = tuple(meta_program_args.FineDiving_key)
    key = (key[0], int(key[1]))
    print(key)

    main(key)
