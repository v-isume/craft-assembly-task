#!/usr/bin/env python

import logging

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import DefaultPredictor, default_setup, default_argument_parser, launch

from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import ema

import numpy as np
import cv2
import os
import torch
import yaml

from collections import defaultdict

#Part type to part color dictionary
classid_to_color = { 
    0: [255, 255, 51], #truck_body
    1: [255, 0, 255],  #truck_cabin
    2: [0, 255, 255],  #wheel
    3: [255, 153, 51], #bus_body
    4: [255, 0, 0],    #furniture_leg
    5: [0, 0, 255],    #chair_seat
    6: [0, 255, 0],    #chair_back
    7: [0, 255, 128],  #chair_arm
    8: [102, 102, 0]   #table_surface
}

#Part type to part color dictionary
classid_to_name = { 
    0: 'truck_body', #
    1: 'truck_cabin',  #
    2: 'wheel',  #
    3: 'bus_body', #
    4: 'furniture_leg',    #
    5: 'chair_seat',    #
    6: 'chair_back',    #
    7: 'chair_arm',  #
    8: 'table_surface'   #
}

class_to_parts = {
    'truck': ['truck_body', 'truck_cabin', 'wheel'],
    'bus': ['bus_body', 'wheel'],
    'chair': ['chair_seat', 'chair_back', 'chair_arm', 'furniture_leg'],
    'table': ['table_surface', 'furniture_leg']
}

classid_to_name_coco = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}


from argparse import Namespace

def get_masks(img_filepath, save_masks=True):
    
    img_name = img_filepath.split('/')[-1]
    img_name = img_name.split('.')[0]

    #parent_path = '/root/home/vitor/'
    output_dir = f'outputs/eva_predictions/{img_name}'
    os.makedirs(output_dir, exist_ok=True)

    #first run EVA02 pretrained on COCO for obj_detection
    config_file = './cfg/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py'
    opts = ['train.init_checkpoint=./weights/eva02_L_coco_det_sys_o365.pth']
    args = Namespace(config_file=config_file, dist_url='tcp://:60900', eval_only=False, machine_rank=1, num_gpus=1, num_machines=1, opts=opts, resume=False, tta=False, wandb=False)
    # load config
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, args)

    # load model
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    # using ema for evaluation
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    # Apply ema state for evaluation
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)

    # change to eval mode
    model.eval()

    # read img
    img = cv2.imread(img_filepath)
    img_tensor = torch.asarray(img)
    img_tensor = torch.moveaxis(img_tensor, -1, 0)

    # change input to Detectron2
    input_dict = defaultdict()
    input_dict["image"] = img_tensor

    # run model
    outputs = model([input_dict])

    # predictions
    pred_classes = outputs[0]['instances'].pred_classes
    pred_scores = outputs[0]['instances'].scores
    pred_boxes = outputs[0]['instances'].pred_boxes.tensor

    coco_thresh = 0.75
    possible_ids = [6, 8, 57, 61]
    top_idx = -1
    crop = False
    major_class = None
    
    # find top scoring mask that belongs to categories
    for i in range(len(pred_scores)):
        pred_class = pred_classes[i].item() + 1
        if pred_class in possible_ids:
            curr_score = pred_scores[i].item()
            if curr_score > coco_thresh:
                coco_thresh = curr_score
                top_idx = i
                crop = True
    
    if crop:
        top_class_id = pred_classes[top_idx].item()
        top_bbox = pred_boxes[top_idx]

        new_img = img[round(top_bbox[1].item()):round(top_bbox[3].item()), round(top_bbox[0].item()):round(top_bbox[2].item())]
        name_class = classid_to_name_coco[top_class_id + 1]
        limited_parts = class_to_parts[name_class]
        if name_class == 'dining table':
            major_class = 'table'
        else:
            major_class = name_class

    else:
        new_img = img
        limited_parts = None

    # free up memory
    del outputs
    del pred_classes
    del pred_scores
    del pred_boxes
    del model
    torch.cuda.empty_cache() 

    # run EVA02 finetuned for our parts
    config_file = './cfg/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8_craft.py'
    opts = ['train.init_checkpoint=./weights/model_final.pth']
    args = Namespace(config_file=config_file, dist_url='tcp://:60900', eval_only=False, machine_rank=1, num_gpus=1, num_machines=1, opts=opts, resume=False, tta=False, wandb=False)
    
    # load config
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, args)

    # load model
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)

    # using ema for evaluation
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    # Apply ema state for evaluation
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)

    # change to eval mode
    model.eval()

    # read img
    img_tensor = torch.asarray(new_img)
    img_tensor = torch.moveaxis(img_tensor, -1, 0)

    # change input to Detectron2
    input_dict = defaultdict()
    input_dict["image"] = img_tensor

    # run model
    outputs = model([input_dict])

    #Extract predictions
    pred_masks = outputs[0]['instances'].pred_masks
    pred_classes = outputs[0]['instances'].pred_classes
    pred_scores = outputs[0]['instances'].scores

    mask_thresh = 0.75

    #Load into the CPU
    masks = pred_masks.cpu().numpy()
    classes = pred_classes.cpu().numpy()
    scores = pred_scores.cpu().detach().numpy()

    counter = 0
    
    # combine masks img
    combined_masks = np.zeros(new_img.shape, np.uint8)
    
    # outputs
    all_masks = []
    all_classes = []

    for mask, class_id, score in zip(masks, classes, scores):
        # check threshold
        if score < mask_thresh:
            continue
        # check limited
        if limited_parts:
            if classid_to_name[class_id] not in limited_parts:
                continue
        
        # create mask for cv2.bitwise_and
        mask_np = np.zeros(new_img.shape[:2], np.uint8)
        mask_np[np.nonzero(mask)] = 255

        # filter the part from the input img
        img_mask = cv2.bitwise_and(new_img, new_img, mask=mask_np)
        non_zero = np.where(
                (img_mask[:, :, 0] != 0) |
                (img_mask[:, :, 1] != 0) |
                (img_mask[:, :, 2] != 0)
        )

        # for visualization, add it to combined_masks
        combined_masks[non_zero] = classid_to_color[class_id]
        
        # output
        all_classes.append(class_id)
        all_masks.append(mask)

        if save_masks:
            #For saving, need to convert the color space
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
            file_name = 'part_' + str(counter) + '.png'
            file_name = os.path.join(output_dir, file_name)
            #Save
            cv2.imwrite(file_name, img_mask)
            counter += 1

    # save combined_masks
    all_masks_save = cv2.cvtColor(combined_masks, cv2.COLOR_RGB2BGR)
    file_name = 'all_combined.png'
    file_name = os.path.join(output_dir, file_name)
    cv2.imwrite(file_name, all_masks_save)
        
    del outputs
    del pred_masks
    del pred_classes
    del pred_scores
    del model
    torch.cuda.empty_cache() 

    if major_class is None:
        if (0 in all_classes) or (1 in all_classes):                          #0: truck_cabin; 1: truck_body
            major_class = 'truck'
        elif (3 in all_classes):                                             #3: bus_body
            major_class = 'bus'
        elif (5 in all_classes) or (6 in all_classes) or (7 in all_classes):   #5: chair_back, 6: chair_seat, 7: chair_arm
            major_class = 'chair'
        elif (8 in all_classes):
            major_class = 'table'      

    # save prediction results in npz file
    np.savez(os.path.join(output_dir, "prediction"), all_masks=all_masks, all_classes=all_classes, major_class=major_class)
    
    return

if __name__ == "__main__":
    with open('./cfg/folders_paths.yaml', 'r') as yaml_file:
        input_folder = yaml.safe_load(yaml_file)['Paths']['input_folder']

    input_list = sorted(os.path.join(input_folder, x) for x in os.listdir(input_folder) if x.lower().endswith('.jpg'))
    for img_filepath in input_list:
        get_masks(img_filepath)