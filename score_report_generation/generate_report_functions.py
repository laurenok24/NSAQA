"""
generate_report_functions.py
Author: Lauren Okamoto
"""

from jinja2 import Environment, FileSystemLoader
import pickle
import os
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import cv2
import base64
from pathlib import Path
import torch
import gradio as gr

############ Generate GIF functions ###################################################################################
def generate_gif(local_directory, image_names, speed_factor=1, loop=0):
    """
    Generate a GIF from a sequence of images paths saved in a local directory.

    Parameters:
    - local_directory (str): Directory path where the images are located
    - image_paths (list): List of filenames of the input images.
    - speed_factor (int): How fast the GIF is, the higher the less the delay is between frames
    - loop (int): Number of loops (0 for infinite loop).

    Returns:
    - Bytes of GIF
    """
    images = []
    durations = []
    for image_name in image_names:
        img = Image.open(os.path.join(local_directory, image_name))
        images.append(img)
        try:
            duration = img.info['duration']
        except KeyError:
            duration = 100  # Default duration in case 'duration' is not available
        durations.append(duration)

    # Calculate the adjusted durations based on the speed factor
    adjusted_durations = [int(duration / speed_factor) for duration in durations]

    # Save as GIF to an in-memory buffer
    gif_buffer = BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True, append_images=images[1:], duration=adjusted_durations, loop=loop)

    # Get the content of the buffer as bytes
    gif_content = gif_buffer.getvalue()
    gif_content = base64.b64encode(gif_content).decode('utf-8')
    return gif_content

def generate_gif_from_frames(frames, speed_factor=1, loop=0, progress=gr.Progress()):
    """
    Generate a GIF from a sequence of images.

    Parameters:
    - frames (list): List of cv2 frames
    - speed_factor (int): How fast the GIF is, the higher the less the delay is between frames
    - loop (int): Number of loops (0 for infinite loop).

    Returns:
    - Bytes of GIF
    """
    durations = []
    images = []
    i = 0
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        images.append(image)
        duration = 100  # Default duration in case 'duration' is not available
        durations.append(duration)
        i+=1

    # Calculate the adjusted durations based on the speed factor
    adjusted_durations = [int(duration / speed_factor) for duration in durations]

    # Save as GIF to an in-memory buffer
    gif_buffer = BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True, append_images=images[1:], duration=adjusted_durations, loop=loop)

    # Get the content of the buffer as bytes
    gif_content = gif_buffer.getvalue()
    gif_content = base64.b64encode(gif_content).decode('utf-8')
    return gif_content

##########################################################################################################
############ Overlay Symbols on Frames  ######################################################################

def draw_pose(keypoints,img, board_end_coord):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    SKELETON = [[1,2],[1,0],[2,6],[3,6],[4,5],[3,4],[6,7],[7,8],[9,8],[7,12],[7,13],[11,12],[13,14],[14,15],[10,11]]
    CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    NUM_KPTS = 16
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_symbols(opencv_image, pose_preds, board_end_coord, plat_outputs, splash_pred_mask, above_board=None):
    if pose_preds is not None:
        draw_pose(np.array(pose_preds)[0],opencv_image, board_end_coord)
    if above_board is None or above_board==1:
        draw_platform(opencv_image, plat_outputs)
    draw_splash(opencv_image, splash_pred_mask)
    return opencv_image

def draw_platform(opencv_image, output):
    pred_classes = output['instances'].pred_classes.cpu().numpy()
    platforms = np.where(pred_classes == 0)[0]
    scores = output['instances'].scores[platforms]
    if len(scores) == 0:
      return
    pred_masks = output['instances'].pred_masks[platforms]
    max_instance = torch.argmax(scores)
    pred_mask = np.array(pred_masks[max_instance].cpu()) 
    # Convert the mask to a binary image    
    binary_mask = pred_mask.squeeze().astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(opencv_image, contours[0], -1, (36, 255, 12), thickness=5)

def draw_splash(opencv_image, pred_mask):
    # Convert the mask to a binary image  
    if pred_mask is None:
        return  
    binary_mask = pred_mask.squeeze().astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(opencv_image, contours[0], -1, (0, 0, 0), thickness=5)

##########################################################################################################
############ Generate HTML template ######################################################################

def generate_report(template_path, data, local_directory, progress=gr.Progress()):
    # Load the template environment
    env = Environment(loader=FileSystemLoader('./score_report_generation/templates'))

    file_names = os.listdir(local_directory)
    file_names.sort()
    file_names = np.array(file_names)
    progress(0.9, desc="Generating Score Report")
    # Load the template
    is_twister = data['twist_position_tightness']['raw_score'] is not None
    overall_score_desc = data['overall_score']['description']
    overall_score = '%.1f' % data['overall_score']['raw_score']

    feet_apart_score = round(float('%.2f' % data['feet_apart']['raw_score']))
    feet_apart_peaks = file_names[data['feet_apart']['peaks']]
    feet_apart_gif = None
    has_feet_apart_peaks = False
    if len(feet_apart_peaks) > 0:
        has_feet_apart_peaks = True
        feet_apart_gif = generate_gif(local_directory, feet_apart_peaks, speed_factor = 0.05)
    feet_apart_percentile = round(float('%.2f' % (data['feet_apart']['percentile'] *100)))
    feet_apart_percentile_divided_by_ten = '%.1f' % (data['feet_apart']['percentile'] *10)

    include_height_off_platform = False
    height_off_board_score = data['height_off_board']['raw_score']
    height_off_board_percentile = data['height_off_board']['percentile']
    height_off_board_description = None
    height_off_board_percentile_divided_by_ten = None
    encoded_height_off_board_frame = None
    if height_off_board_score is not None:
        include_height_off_platform = True
        height_off_board_score = round(float('%.2f' % data['height_off_board']['raw_score']))
        height_off_board_frame = file_names[data['height_off_board']['frame_index']]
        height_off_board_frame_path = os.path.join(local_directory, height_off_board_frame)
        with open(height_off_board_frame_path, "rb") as image_file:
            encoded_height_off_board_frame = base64.b64encode(image_file.read()).decode('utf-8')
        height_off_board_percentile = round(float('%.2f' % (data['height_off_board']['percentile'] *100)))
        height_off_board_percentile_divided_by_ten = '%.1f' % (data['height_off_board']['percentile'] *10)
        if float(height_off_board_percentile_divided_by_ten) > 5:
            height_off_board_description = "good"
        else:
            height_off_board_description = "a bit on the lower side"

    dist_from_board_score = '%.2f' % data['distance_from_board']['raw_score']
    dist_from_board_frame = file_names[data['distance_from_board']['frame_index']]
    dist_from_board_frame_path = os.path.join(local_directory, dist_from_board_frame)
    with open(dist_from_board_frame_path, "rb") as image_file:
        encoded_dist_from_board_frame = base64.b64encode(image_file.read()).decode('utf-8')
    dist_from_board_percentile = data['distance_from_board']['percentile']
    if 'good' in dist_from_board_percentile:
        dist_from_board_percentile_status = "Good"
    elif 'far' in dist_from_board_percentile:
        dist_from_board_percentile_status = "Too Far"
    else:
        dist_from_board_percentile_status = "Too Close"

    knee_bend_score = data['knee_bend']['raw_score']
    knee_bend_percentile = data['knee_bend']['percentile']
    knee_bend_frames = []
    knee_bend_percentile_divided_by_ten = None
    if knee_bend_score is not None:
        knee_bend_score = round(float('%.2f' % (data['knee_bend']['raw_score'])))
        knee_bend_frames = file_names[data['knee_bend']['frame_indices']]
        knee_bend_percentile = round(float('%.2f' % (knee_bend_percentile * 100)))
        knee_bend_percentile_divided_by_ten = '%.1f' % (data['knee_bend']['percentile'] * 10)
    
    som_position_tightness_score = data['som_position_tightness']['raw_score']
    som_position_tightness_percentile = data['som_position_tightness']['percentile']
    som_position_tightness_position = data['som_position_tightness']['position']
    som_position_tightness_frames = file_names[data['som_position_tightness']['frame_indices']]
    som_position_tightness_gif = None
    som_position_tightness_percentile_divided_by_ten = None
    if som_position_tightness_score is not None:
        if is_twister:
            som_position_tightness_score = round(float('%.2f' % (data['som_position_tightness']['raw_score'] + 15)))
        else:
            som_position_tightness_score = round(float('%.2f' % (data['som_position_tightness']['raw_score'])))
        som_position_tightness_gif = generate_gif(local_directory, som_position_tightness_frames)
        som_position_tightness_percentile = round(float('%.2f' % (som_position_tightness_percentile * 100)))
        som_position_tightness_percentile_divided_by_ten = '%.1f' % (data['som_position_tightness']['percentile'] * 10)
    twist_position_tightness_score = data['twist_position_tightness']['raw_score']
    twist_position_tightness_frames = []
    twist_position_tightness_gif = None
    twist_position_tightness_percentile = None
    twist_position_tightness_percentile_divided_by_ten = None
    if twist_position_tightness_score is not None:
        twist_position_tightness_score = round(float('%.2f' % twist_position_tightness_score))
        twist_position_tightness_frames = file_names[data['twist_position_tightness']['frame_indices']]
        twist_position_tightness_gif = generate_gif(local_directory, twist_position_tightness_frames)
        twist_position_tightness_percentile = round(float('%.2f' % (data['twist_position_tightness']['percentile'] * 100)))
        twist_position_tightness_percentile_divided_by_ten = '%.1f' % (data['twist_position_tightness']['percentile'] * 10)
    over_under_rotation_score = round(float('%.2f' % data['over_under_rotation']['raw_score']))
    over_under_rotation_frame = file_names[data['over_under_rotation']['frame_index']]
    over_under_rotation_percentile = round(float('%.2f' % (data['over_under_rotation']['percentile'] * 100)))
    over_under_rotation_percentile_divided_by_ten = '%.1f' % (data['over_under_rotation']['percentile'] * 10)
    straightness_during_entry_score = round(float('%.2f' % data['straightness_during_entry']['raw_score']))
    straightness_during_entry_frames = file_names[data['straightness_during_entry']['frame_indices']]
    straightness_during_entry_gif = generate_gif(local_directory, straightness_during_entry_frames, speed_factor = 0.5)
    straightness_during_entry_percentile = round(float('%.2f' % (data['straightness_during_entry']['percentile'] * 100)))
    straightness_during_entry_percentile_divided_by_ten = '%.1f' % (data['straightness_during_entry']['percentile'] * 10)
    splash_score = round(float('%.2f' % data['splash']['raw_score']))
    splash_frame = file_names[data['splash']['maximum_index']]
    splash_indices = file_names[data['splash']['frame_indices']]
    splash_gif = None
    if len(splash_indices) > 0:
        splash_gif = generate_gif(local_directory, splash_indices)
    splash_percentile = round(float('%.2f' % (data['splash']['percentile'] * 100)))
    splash_percentile_divided_by_ten = '%.1f' % (data['splash']['percentile'] * 10)
    if float(splash_percentile) < 50:
        splash_description = 'on the larger side'
    else:
        splash_description = 'small'
    template = env.get_template(template_path)
    data = {
        'local_directory': local_directory,
        'is_twister' : is_twister,
        'overall_score_desc' : overall_score_desc,
        'overall_score' : overall_score,
        'feet_apart_score' : feet_apart_score,
        'feet_apart_peaks' : feet_apart_peaks,
        'has_feet_apart_peaks' : has_feet_apart_peaks,
        'feet_apart_gif' : feet_apart_gif,
        'feet_apart_percentile' : feet_apart_percentile,
        'feet_apart_percentile_divided_by_ten': feet_apart_percentile_divided_by_ten,

        'include_height_off_platform': include_height_off_platform,
        'height_off_board_score' : height_off_board_score,
        'height_off_board_percentile' : height_off_board_percentile,
        'encoded_height_off_board_frame' : encoded_height_off_board_frame,
        'height_off_board_percentile_divided_by_ten': height_off_board_percentile_divided_by_ten,
        'height_off_board_description' : height_off_board_description,

        'dist_from_board_score' : dist_from_board_score,
        'dist_from_board_frame' : dist_from_board_frame,
        'encoded_dist_from_board_frame': encoded_dist_from_board_frame,
        'dist_from_board_percentile' : dist_from_board_percentile,
        'dist_from_board_percentile_status': dist_from_board_percentile_status,

        'knee_bend_score' : knee_bend_score,
        'knee_bend_frames' : knee_bend_frames,
        'knee_bend_percentile' : knee_bend_percentile,
        'knee_bend_percentile_divided_by_ten' : knee_bend_percentile_divided_by_ten,

        'som_position_tightness_score' : som_position_tightness_score,
        'som_position_tightness_frames' : som_position_tightness_frames,
        'som_position_tightness_gif' : som_position_tightness_gif,
        'som_position_tightness_position' : som_position_tightness_position,
        'som_position_tightness_percentile' : som_position_tightness_percentile,
        'som_position_tightness_percentile_divided_by_ten' : som_position_tightness_percentile_divided_by_ten,
        'twist_position_tightness_score' : twist_position_tightness_score,
        'twist_position_tightness_frames': twist_position_tightness_frames,
        'twist_position_tightness_percentile' : twist_position_tightness_percentile,
        'twist_position_tightness_percentile_divided_by_ten' : twist_position_tightness_percentile_divided_by_ten,
        'twist_position_tightness_gif' : twist_position_tightness_gif,
        'over_under_rotation_score' : over_under_rotation_score,
        'over_under_rotation_frame' : over_under_rotation_frame,
        'over_under_rotation_percentile' : over_under_rotation_percentile,
        'over_under_rotation_percentile_divided_by_ten' : over_under_rotation_percentile_divided_by_ten,
        'straightness_during_entry_score' : straightness_during_entry_score,
        'straightness_during_entry_gif' : straightness_during_entry_gif,
        'straightness_during_entry_percentile' : straightness_during_entry_percentile,
        'straightness_during_entry_percentile_divided_by_ten': straightness_during_entry_percentile_divided_by_ten,
        'splash_score' : splash_score,
        'splash_frame' : splash_frame,
        'splash_gif' : splash_gif,
        'splash_percentile' : splash_percentile,
        'splash_percentile_divided_by_ten': splash_percentile_divided_by_ten,
        'splash_description' : splash_description,
    }
    # Render the template with the provided data
    report_content = template.render(data)
    return report_content


def generate_report_from_frames(template_path, data, frames):
    # Load the template environment
    env = Environment(loader=FileSystemLoader('./score_report_generation/templates'))

    frames = np.array(frames)
    # Load the template
    is_twister = data['twist_position_tightness']['raw_score'] is not None
    overall_score_desc = data['overall_score']['description']
    overall_score = '%.1f' % data['overall_score']['raw_score']

    feet_apart_score = round(float('%.2f' % data['feet_apart']['raw_score']))
    feet_apart_peaks = frames[data['feet_apart']['peaks']]
    feet_apart_gif = None
    has_feet_apart_peaks = False
    if len(feet_apart_peaks) > 0:
        has_feet_apart_peaks = True
        feet_apart_gif = generate_gif_from_frames(feet_apart_peaks, speed_factor = 0.05)
    feet_apart_percentile = round(float('%.2f' % (data['feet_apart']['percentile'] *100)))
    feet_apart_percentile_divided_by_ten = '%.1f' % (data['feet_apart']['percentile'] *10)

    include_height_off_platform = False
    height_off_board_score = data['height_off_board']['raw_score']
    height_off_board_frame = None
    height_off_board_percentile = None
    height_off_board_percentile_divided_by_ten = None
    height_off_board_description = None
    encoded_height_off_board_frame = None
    if height_off_board_score is not None:
        include_height_off_platform = True
        height_off_board_score = round(float('%.2f' % data['height_off_board']['raw_score']))
        height_off_board_frame = Image.fromarray(cv2.cvtColor(frames[data['height_off_board']['frame_index']], cv2.COLOR_BGR2RGB))
        height_buffer = BytesIO()
        height_off_board_frame.save(height_buffer, format='JPEG')
        encoded_height_off_board_frame = base64.b64encode(height_buffer.getvalue()).decode('utf-8')
        height_off_board_percentile = round(float('%.2f' % (data['height_off_board']['percentile'] *100)))
        height_off_board_percentile_divided_by_ten = '%.1f' % (data['height_off_board']['percentile'] *10)
        if float(height_off_board_percentile_divided_by_ten) > 5:
            height_off_board_description = "good"
        else:
            height_off_board_description = "a bit on the lower side"

    dist_from_board_score = '%.2f' % data['distance_from_board']['raw_score']
    dist_from_board_frame = Image.fromarray(cv2.cvtColor(frames[data['distance_from_board']['frame_index']], cv2.COLOR_BGR2RGB))
    dist_buffer = BytesIO()
    dist_from_board_frame.save(dist_buffer, format='JPEG')
    encoded_dist_from_board_frame = base64.b64encode(dist_buffer.getvalue()).decode('utf-8')
    dist_from_board_percentile = data['distance_from_board']['percentile']
    if 'good' in dist_from_board_percentile:
        dist_from_board_percentile_status = "Good"
    elif 'far' in dist_from_board_percentile:
        dist_from_board_percentile_status = "Too Far"
    else:
        dist_from_board_percentile_status = "Too Close"

    knee_bend_score = data['knee_bend']['raw_score']
    knee_bend_percentile = data['knee_bend']['percentile']
    knee_bend_frames = []
    knee_bend_percentile_divided_by_ten = None
    if knee_bend_score is not None:
        knee_bend_score = round(float('%.2f' % knee_bend_score))
        knee_bend_percentile = round(float('%.2f' % (knee_bend_percentile * 100)))
        knee_bend_frames = frames[data['knee_bend']['frame_indices']]
        knee_bend_percentile_divided_by_ten = '%.1f' % (data['knee_bend']['percentile'] * 10)

    som_position_tightness_score = data['som_position_tightness']['raw_score']
    som_position_tightness_percentile = data['som_position_tightness']['percentile']
    som_position_tightness_position = data['som_position_tightness']['position']
    som_position_tightness_frames = []
    som_position_tightness_gif = None
    som_position_tightness_percentile_divided_by_ten = None
    if som_position_tightness_score is not None:
        if is_twister:
            som_position_tightness_score = round(float('%.2f' % (data['som_position_tightness']['raw_score'] + 15)))
        else:
            som_position_tightness_score = round(float('%.2f' % (data['som_position_tightness']['raw_score'])))
        som_position_tightness_frames = frames[data['som_position_tightness']['frame_indices']]
        som_position_tightness_gif = generate_gif_from_frames(som_position_tightness_frames)
        som_position_tightness_percentile = round(float('%.2f' % (som_position_tightness_percentile * 100)))
        som_position_tightness_percentile_divided_by_ten = '%.1f' % (data['som_position_tightness']['percentile'] * 10)

    twist_position_tightness_score = data['twist_position_tightness']['raw_score']
    twist_position_tightness_frames = []
    twist_position_tightness_gif = None
    twist_position_tightness_percentile = None
    twist_position_tightness_percentile_divided_by_ten = None
    if twist_position_tightness_score is not None:
        twist_position_tightness_score = round(float('%.2f' % twist_position_tightness_score))
        twist_position_tightness_frames = frames[data['twist_position_tightness']['frame_indices']]
        twist_position_tightness_gif = generate_gif_from_frames(twist_position_tightness_frames)
        twist_position_tightness_percentile = round(float('%.2f' % (data['twist_position_tightness']['percentile'] * 100)))
        twist_position_tightness_percentile_divided_by_ten = '%.1f' % (data['twist_position_tightness']['percentile'] * 10)

    over_under_rotation_score = round(float('%.2f' % data['over_under_rotation']['raw_score']))
    over_under_rotation_frame = frames[data['over_under_rotation']['frame_index']]
    over_under_rotation_percentile = round(float('%.2f' % (data['over_under_rotation']['percentile'] * 100)))
    over_under_rotation_percentile_divided_by_ten = '%.1f' % (data['over_under_rotation']['percentile'] * 10)

    straightness_during_entry_score = round(float('%.2f' % data['straightness_during_entry']['raw_score']))
    straightness_during_entry_frames = frames[data['straightness_during_entry']['frame_indices']]
    straightness_during_entry_gif = generate_gif_from_frames(straightness_during_entry_frames, speed_factor = 0.5)
    straightness_during_entry_percentile = round(float('%.2f' % (data['straightness_during_entry']['percentile'] * 100)))
    straightness_during_entry_percentile_divided_by_ten = '%.1f' % (data['straightness_during_entry']['percentile'] * 10)

    splash_score = round(float('%.2f' % data['splash']['raw_score']))
    splash_frame = frames[data['splash']['maximum_index']]
    splash_indices = frames[data['splash']['frame_indices']]
    splash_gif = None
    if len(splash_indices) > 0:
        splash_gif = generate_gif_from_frames(splash_indices)
    splash_percentile = round(float('%.2f' % (data['splash']['percentile'] * 100)))
    splash_percentile_divided_by_ten = '%.1f' % (data['splash']['percentile'] * 10)
    if float(splash_percentile) < 50:
        splash_description = 'on the larger side'
    else:
        splash_description = 'small'
    template = env.get_template(template_path)
    data = {
        'is_twister' : is_twister,
        'overall_score_desc' : overall_score_desc,
        'overall_score' : overall_score,
        'feet_apart_score' : feet_apart_score,
        'feet_apart_peaks' : feet_apart_peaks,
        'has_feet_apart_peaks' : has_feet_apart_peaks,
        'feet_apart_gif' : feet_apart_gif,
        'feet_apart_percentile' : feet_apart_percentile,
        'feet_apart_percentile_divided_by_ten': feet_apart_percentile_divided_by_ten,
        'include_height_off_platform': include_height_off_platform,
        'height_off_board_score' : height_off_board_score,
        'height_off_board_percentile' : height_off_board_percentile,
        'encoded_height_off_board_frame' : encoded_height_off_board_frame,
        'height_off_board_percentile_divided_by_ten': height_off_board_percentile_divided_by_ten,
        'height_off_board_description' : height_off_board_description,
        'dist_from_board_score' : dist_from_board_score,
        'dist_from_board_frame' : dist_from_board_frame,
        'encoded_dist_from_board_frame': encoded_dist_from_board_frame,
        'dist_from_board_percentile' : dist_from_board_percentile,
        'dist_from_board_percentile_status': dist_from_board_percentile_status,
        'knee_bend_score' : knee_bend_score,
        'knee_bend_frames' : knee_bend_frames,
        'knee_bend_percentile' : knee_bend_percentile,
        'knee_bend_percentile_divided_by_ten' : knee_bend_percentile_divided_by_ten,
        'som_position_tightness_score' : som_position_tightness_score,
        'som_position_tightness_frames' : som_position_tightness_frames,
        'som_position_tightness_gif' : som_position_tightness_gif,
        'som_position_tightness_position' : som_position_tightness_position,
        'som_position_tightness_percentile' : som_position_tightness_percentile,
        'som_position_tightness_percentile_divided_by_ten' : som_position_tightness_percentile_divided_by_ten,
        'twist_position_tightness_score' : twist_position_tightness_score,
        'twist_position_tightness_frames': twist_position_tightness_frames,
        'twist_position_tightness_percentile' : twist_position_tightness_percentile,
        'twist_position_tightness_percentile_divided_by_ten' : twist_position_tightness_percentile_divided_by_ten,
        'twist_position_tightness_gif' : twist_position_tightness_gif,
        'over_under_rotation_score' : over_under_rotation_score,
        'over_under_rotation_frame' : over_under_rotation_frame,
        'over_under_rotation_percentile' : over_under_rotation_percentile,
        'over_under_rotation_percentile_divided_by_ten' : over_under_rotation_percentile_divided_by_ten,
        'straightness_during_entry_score' : straightness_during_entry_score,
        'straightness_during_entry_gif' : straightness_during_entry_gif,
        'straightness_during_entry_percentile' : straightness_during_entry_percentile,
        'straightness_during_entry_percentile_divided_by_ten': straightness_during_entry_percentile_divided_by_ten,
        'splash_score' : splash_score,
        'splash_frame' : splash_frame,
        'splash_gif' : splash_gif,
        'splash_percentile' : splash_percentile,
        'splash_percentile_divided_by_ten': splash_percentile_divided_by_ten,
        'splash_description' : splash_description,
    }
    # Render the template with the provided data
    report_content = template.render(data)
    return report_content

def generate_symbols_report(template_path, dive_data, frames):
    # Load the template environment
    env = Environment(loader=FileSystemLoader('./score_report_generation/templates'))
    template = env.get_template(template_path)
    pose_frames = []
    for i in range(len(dive_data['pose_pred'])):
        pose_frame = draw_symbols(frames[i], dive_data['pose_pred'][i],  dive_data['board_end_coords'][i], dive_data['plat_outputs'][i], dive_data['splash_pred_masks'][i])
        pose_frames.append(pose_frame)
    pose_gif = generate_gif_from_frames(pose_frames, speed_factor=2)
    pose_data = {}
    pose_data['pose_gif'] = pose_gif
    html = template.render(pose_data)
    return html

def generate_symbols_report_precomputed(template_path, dive_data, local_directory, progress=gr.Progress()):
    # Load the template environment
    file_names = os.listdir(local_directory)
    file_names.sort()
    file_names = np.array(file_names)

    if 'above_boards' in dive_data:
        above_boards = dive_data['above_boards']
    else:
        above_boards = [None] * len(file_names)

    env = Environment(loader=FileSystemLoader('./score_report_generation/templates'))
    template = env.get_template(template_path)
    pose_frames = []
    counter = 0
    for i in range(len(file_names)):
        progress(i/(len(file_names)+10), desc="Abstracting Symbols")
        if file_names[i][-4:] != ".jpg":
            continue
        opencv_image = cv2.imread(local_directory+file_names[i])
        pose_frame = draw_symbols(opencv_image, dive_data['pose_pred'][counter],  dive_data['board_end_coords'][counter], dive_data['plat_outputs'][counter], dive_data['splash_pred_masks'][counter], above_board=above_boards[counter])
        pose_frames.append(pose_frame)
        counter +=1
    pose_gif = generate_gif_from_frames(pose_frames, speed_factor=2, progress=progress)
    pose_data = {}
    pose_data['pose_gif'] = pose_gif
    html = template.render(pose_data)
    return html
