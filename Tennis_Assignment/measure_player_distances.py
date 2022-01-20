#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 19:01:33 2021

@author: keshav
"""

import cv2

from PIL import Image
import imagehash
import numpy as np

import os

from yolov5_predictor import predict_bbs_and_return2

import math
import cv2
from time import time
import matplotlib.pyplot as plt

from imutils import perspective

#%%

# from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
# from functools import partial
# import itertools
from itertools import repeat
import multiprocessing
# from itertools import product
# from io import BytesIO


from PIL import Image

NUM_WORKERS = min(8, mp.cpu_count()-2)
# TARGET_SIZE = 512  # image resolution to be stored
# IMG_QUALITY = 90  # JPG quality
print("Number of workers is {}".format(NUM_WORKERS))

#%%

def save_twice_warped_image(x):
    file_name,warp1_pts,warp2_pts,crop_file_name = x
    img = cv2.imread(file_name)
    warp1 = perspective.four_point_transform(img, warp1_pts)
    warp2 = perspective.four_point_transform(warp1, warp2_pts)
    cv2.imwrite(crop_file_name, warp2)

#%%

# Loading the reference court extreme points
import pickle
with open('court_points_sorted.pkl','rb') as f:
    court_points_sorted = pickle.load(f)

    
#%%%%%

# Expanding the court points to capture the players playing outside and on the court boundary
# Generating the 4 points to perform the 1st perspective transform

extended_court_points = []
for i,p in enumerate(court_points_sorted):
    if i==0:
        p2 = (int(p[0]/1.60),int(p[1]*1.4))
    elif i==1:
        p2 = (int(p[0]/1.60),int(p[1]/1.85))
    elif i==2:
        p2 = (int(p[0]*1.15),int(p[1]/1.85))
    elif i==3:
        p2 = (int(p[0]*1.15),int(p[1]*1.4))     
    extended_court_points.append(p2)
    
warp1_pts = np.array([extended_court_points[i] for i in [1,2,3,0]])    

# Court points, obtained after manual inspection on the result after 1st perspective transform
warp2_pts = np.array([(0, 660), (243, 1), (908, 1), (1070, 660)])

#%%

save_twice_warped_image(['court_reference.jpg',
                         warp1_pts,
                         warp2_pts,
                         'warped_court.jpg']
                         )  

#%%

"""
Tennis court has width of 10.97 metres, or 11 metres approximately. 
These 11 metres are spread over 1000 pixels in the warped court image.

Similarly,
Tennis court has length of 23.78 metres, spread over 490 pixels in the warped court image.
"""

x_axis_mpp = 11/1000
y_axis_mpp = 23.78/490

#%%

selected_frames_directory = 'selected_frames_60fps'
warped_frames_directory = 'warped_frames_60fps'

if not os.path.exists(warped_frames_directory):
    os.makedirs(warped_frames_directory)    

#%%

all_frames = os.listdir(selected_frames_directory)


frame_nums = [int(frame.split('_')[1]) for frame in all_frames]

frameNum_frameName = sorted(zip(frame_nums,all_frames))

num_name_dict = dict(frameNum_frameName)

#%%

# Split video to clips

ini_frameNum = 0

video_clips_ending_frames = []

video_clip_frames_list = []
frame_list = []

for i,(frame_num,file_name) in enumerate(frameNum_frameName):
    
    if frame_num-ini_frameNum>60:
        video_clips_ending_frames.append(frame_num)
        ini_frameNum = frameNum_frameName[i][0]
        video_clip_frames_list.append(frame_list)
        frame_list = []
    else:
        ini_frameNum = frame_num
        frame_list.append(frame_num)

#%%

# Creating the filenames to load the selected frames, and save path

frame_filepaths = []
warped_filepaths = []


for i,clip_ending in enumerate(video_clips_ending_frames):
    
    clip_files = []
    # clip_processed_files = []
    clip_warped_files = []
    
    for frame_idx in video_clip_frames_list[i]:
        
        frame_filename = num_name_dict[frame_idx]
        frame_filepath = '{}/{}'.format(selected_frames_directory,frame_filename)
        print('Processing {}'.format(frame_filename))
        warped_filepath = '{}/{}'.format(warped_frames_directory,frame_filename)
        
        clip_files.append(frame_filepath)
        clip_warped_files.append(warped_filepath)

    frame_filepaths.append(clip_files)
    warped_filepaths.append(clip_warped_files)
   
#%%

selected_clips = [i for i,x in enumerate(frame_filepaths) if len(x)>9]
frame_filepaths2 = [frame_filepaths[i] for i in selected_clips]
warped_filepaths2 = [warped_filepaths[i] for i in selected_clips]

#%%

# Perform perspective transformation twice on the selected frames with extended court coordinates
                
for i,file_paths in enumerate(frame_filepaths2):
    # if i>0:
        # break
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        result = pool.map(save_twice_warped_image,list(zip(file_paths,
                                                      repeat(warp1_pts),
                                                      repeat(warp2_pts),
                                                      warped_filepaths2[i])))

#%%        

"""
Record Player Coordinates Clipwise 
"""

# Yolo Detects multiple persons at a time, setting a reference to exclude non-playing people standing on outskirts
player1_avg_y_coordinate = 188
player2_avg_y_coordinate = 670

# List of list, to capture the player wise coordinates, in each clip
player1_clip_coords = []
player2_clip_coords = []


clip_frames_skipped = []

for i,clip_ending in enumerate(frame_filepaths2):
    
    # if i!=0:
    #     continue
    
    p1_coords = []
    p2_coords = []
    frames_skipped = []
    
    for warped_file in warped_filepaths2[i]:
        
        print('Processing {}'.format(warped_file))
        
        image_captions,bb_coords,v_scores,actual_img_size,middle_coords = predict_bbs_and_return2(warped_file)
        
        person_idxs = [i for i,x in enumerate(image_captions) if x=='person']
        ball_idxs = [i for i,x in enumerate(image_captions) if x=='sports ball']

        if image_captions.count('person')<2:
            frames_skipped.append(warped_file)
            continue        
        
        person_middle_coords = [middle_coords[i] for i in person_idxs]
    
        middle_coords_player1 = [abs(y[1]-player1_avg_y_coordinate) for y in person_middle_coords]
        middle_coords_player2 = [abs(y[1]-player2_avg_y_coordinate) for y in person_middle_coords]
        
        player1_coord_idx = np.argmin(middle_coords_player1)
        player2_coord_idx = np.argmin(middle_coords_player2)
        

        p1_pt = person_middle_coords[player1_coord_idx]
        if p1_pt[1] in range(player1_avg_y_coordinate-100,player1_avg_y_coordinate+100):
            p1_coords.append((warped_file,p1_pt))
        
        p2_pt = person_middle_coords[player2_coord_idx]
        if p2_pt[1] in range(player2_avg_y_coordinate-100,player2_avg_y_coordinate+100):
            p2_coords.append((warped_file,p2_pt))
    
    
    player1_clip_coords.append(p1_coords)
    player2_clip_coords.append(p2_coords)
    clip_frames_skipped.append(frames_skipped)
    
    print('\nClip {} ENDED\n'.format(i+1))

#%%

def ret_euclidean(p1,p2,x_axis_mpp=1,y_axis_mpp=1):
    return ((x_axis_mpp*(p1[0]-p2[0]))**2+(y_axis_mpp*(p1[1]-p2[1]))**2)**0.5

#%%

player1_clip_dists = []
for i,clip_coords in enumerate(player1_clip_coords):
    ini = clip_coords[0][1]
    clip_distance = 0
    for x in range(1,len(clip_coords)):
        clip_distance = clip_distance + ret_euclidean(ini, clip_coords[x][1],x_axis_mpp,y_axis_mpp)
        ini = clip_coords[x][1]
    player1_clip_dists.append(clip_distance)
    
player1_total_distance = sum(player1_clip_dists)
    
#%%

player2_clip_dists = []
for i,clip_coords in enumerate(player2_clip_coords):
    ini = clip_coords[0][1]
    clip_distance = 0
    for x in range(1,len(clip_coords)):
        clip_distance = clip_distance + ret_euclidean(ini, clip_coords[x][1],x_axis_mpp,y_axis_mpp)
        ini = clip_coords[x][1]
    player2_clip_dists.append(clip_distance)
    
player2_total_distance = sum(player2_clip_dists)

#%%

print('Player 1 travelled: {} metres'.format(player1_total_distance))
print('Player 2 travelled: {} metres'.format(player2_total_distance))

#%%


video_clips_longer_than_1_sec = [i for i,x in enumerate(video_clip_frames_list) if len(x)>=60]

print("Total number of true serves made: {}".format(len(video_clips_longer_than_1_sec)))

#%%

# Filtering clips with duration longer than 1 second, to shortlist actual clips, AND
# Merging actual clips with break less than 180 frames or atleast 3 seconds apart 
# to find the least number of true serves

actual_clips = 0
ini_clip_end = video_clip_frames_list[0][-1]
for clip_frames in video_clip_frames_list:
    if len(clip_frames)>=60:
        clip_start = clip_frames[0]
        if clip_start-ini_clip_end>180:
            actual_clips = actual_clips+1
        ini_clip_end = clip_frames[-1]
        
print("Least number of true serves made: {}".format(actual_clips))        

#%%