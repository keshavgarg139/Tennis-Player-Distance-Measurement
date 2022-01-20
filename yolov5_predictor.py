#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:38:35 2021

@author: keshav
"""

#%%

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from datetime import datetime

#%%

from PIL import Image
import os

#%%

from models.experimental import attempt_load

#%%

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

#%%

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

#%%

# object_detection_model_directory = model_directory+'/object-detection-model'

device = select_device('cpu')
half = device.type != 'cpu'
model = attempt_load('yolov5l.pt', map_location=device)
stride = int(model.stride.max())
cudnn.benchmark = True
imgsz = check_img_size(640, s=stride)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
print(names)

def predict_bbs_and_save(input_img_path,bounding_box_directory):
    
    #%%
    
    frame=cv2.imread(input_img_path)
    
    img=frame.copy()
    gframe=frame.copy()
     
    img = letterbox(img,stride=32)[0]
    img = np.stack(img, 0)

            # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img=img.float()
    img /=255.0
    if img.ndimension() == 3:
              img = img.unsqueeze(0)
    pred = model(img,augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)


    actual_img = Image.open(input_img_path)
    actual_img_size = actual_img.size
    photo_name = input_img_path.split('/')[-1].split('.')[0].split('_')[0]    
    
    image_paths = []
    image_captions = []
    bb_coords = []
    v_scores = []
    # v_labels = []
    
    bb_count = 0
    
    for i, det in enumerate(pred):

      det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
      bbxywh=[]
      clsconf=[]
     
      for *xyxy, conf, cls in reversed(det):
       
        if conf>0.1:
            a=int((xyxy[0]))
            b=int((xyxy[1]))
            c=int((xyxy[2]))
            d=int((xyxy[3]))
            cx=int((a+c)/2)
            cy=int((b+d)/2)
            x=int((xyxy[0]+xyxy[2])/2)
            y=int((xyxy[1]+xyxy[3])/2)
            w=2*(x-a)
            h=2*(y-b)
            # print(x,y,w,h)
            bbxywh.append([x,y,w,h])
            clsconf.append(conf)
            start=(a,b)
            color=(255,0,0)
            end=(c,d)
            
            label = names[int(cls)]
            
            if label not in yoloV5Caption_to_subcat_mapping.keys():
                continue
            
            bb_filename = photo_name+'_bb{}_{}.jpg'.format(bb_count+1,label)
            bb_filename2 = os.path.join(bounding_box_directory,bb_filename)
            
            cropped_img = actual_img.crop((a,b,c,d))
            cropped_img.save(bb_filename2)
            
            print([(a,c),(b,d)],[x,y],[w,h],conf,label)
            
            image_paths.append(bb_filename2)
            image_captions.append(label)
            bb_coords.append([(a,b),(c,d)])
            v_scores.append(float(conf))
            
            bb_count = bb_count+1
    
    #%%
            
    return image_paths,image_captions,bb_coords,v_scores,actual_img_size

#%%

def predict_bbs_and_return(input_img_path):
    
    #%%
    
    frame=cv2.imread(input_img_path)
    
    img=frame.copy()
    gframe=frame.copy()
     
    img = letterbox(img,stride=32)[0]
    img = np.stack(img, 0)

            # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img=img.float()
    img /=255.0
    if img.ndimension() == 3:
              img = img.unsqueeze(0)
    pred = model(img,augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)


    actual_img = Image.open(input_img_path)
    actual_img_size = actual_img.size
    photo_name = input_img_path.split('/')[-1].split('.')[0].split('_')[0]    
    
    # image_paths = []
    image_captions = []
    bb_coords = []
    v_scores = []
    middle_coords = []
    # v_labels = []
    
    bb_count = 0
    
    for i, det in enumerate(pred):

      det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
      bbxywh=[]
      clsconf=[]
     
      for *xyxy, conf, cls in reversed(det):
       
        if conf>0.1:
            a=int((xyxy[0]))
            b=int((xyxy[1]))
            c=int((xyxy[2]))
            d=int((xyxy[3]))
            cx=int((a+c)/2)
            cy=int((b+d)/2)
            x=int((xyxy[0]+xyxy[2])/2)
            y=int((xyxy[1]+xyxy[3])/2)
            w=2*(x-a)
            h=2*(y-b)
            # print(x,y,w,h)
            bbxywh.append([x,y,w,h])
            clsconf.append(conf)
            start=(a,b)
            color=(255,0,0)
            end=(c,d)
            
            label = names[int(cls)]
            
            if label!='person':
                continue
            
            # bb_filename = photo_name+'_bb{}_{}.jpg'.format(bb_count+1,label)
            # bb_filename2 = os.path.join(bounding_box_directory,bb_filename)
            
            # cropped_img = actual_img.crop((a,b,c,d))
            # cropped_img.save(bb_filename2)
            
            # print([(a,c),(b,d)],[x,y],[w,h],conf,label)
            
            # image_paths.append(bb_filename2)
            image_captions.append(label)
            bb_coords.append([(a,b),(c,d)])
            v_scores.append(float(conf))
            middle_coords.append([int((a+c)/2),d])
            
            bb_count = bb_count+1
    
    #%%
            
    return image_captions,bb_coords,v_scores,actual_img_size,middle_coords

def predict_bbs_and_return2(input_img_path):
    
    #%%
    
    frame=cv2.imread(input_img_path)
    
    img=frame.copy()
    gframe=frame.copy()
     
    img = letterbox(img,stride=32)[0]
    img = np.stack(img, 0)

            # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img=img.float()
    img /=255.0
    if img.ndimension() == 3:
              img = img.unsqueeze(0)
    pred = model(img,augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)


    actual_img = Image.open(input_img_path)
    actual_img_size = actual_img.size
    photo_name = input_img_path.split('/')[-1].split('.')[0].split('_')[0]    
    
    # image_paths = []
    image_captions = []
    bb_coords = []
    v_scores = []
    middle_coords = []
    # v_labels = []
    
    bb_count = 0
    
    for i, det in enumerate(pred):

      det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
      bbxywh=[]
      clsconf=[]
     
      for *xyxy, conf, cls in reversed(det):
       
        if conf>0.1:
            a=int((xyxy[0]))
            b=int((xyxy[1]))
            c=int((xyxy[2]))
            d=int((xyxy[3]))
            cx=int((a+c)/2)
            cy=int((b+d)/2)
            x=int((xyxy[0]+xyxy[2])/2)
            y=int((xyxy[1]+xyxy[3])/2)
            w=2*(x-a)
            h=2*(y-b)
            # print(x,y,w,h)
            bbxywh.append([x,y,w,h])
            clsconf.append(conf)
            start=(a,b)
            color=(255,0,0)
            end=(c,d)
            
            label = names[int(cls)]
            
            if label not in ['person','sports ball']:
                continue
            
            image_captions.append(label)
            bb_coords.append([(a,b),(c,d)])
            v_scores.append(float(conf))
            middle_coords.append([int((a+c)/2),d])
            
            bb_count = bb_count+1
    
    #%%
            
    return image_captions,bb_coords,v_scores,actual_img_size,middle_coords