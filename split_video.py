#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:14:45 2021

@author: keshav
"""


#%%

# loading the required libraries

import cv2

from PIL import Image
import imagehash

def BGR_to_RGB(image_array):
    destRGB = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return destRGB

#%%

# loading the video

vidcap = cv2.VideoCapture('tennis_video_assignment.mp4')

fps = vidcap.get(cv2.CAP_PROP_FPS)

print('Actual FPS of video: {}'.format(int(fps)))

#%%

from moviepy.editor import *
    
# loading video dsa gfg intro video 
clip = VideoFileClip("tennis_video_assignment.mp4") 
     
# getting only first 5 seconds
# clip = clip.subclip(0, 5)
  
# new clip with new fps
new_clip = clip.set_fps(60)
    
#%%

# Saving the downsampled video
 
required_fps = 60

new_clip.write_videofile("tennis_video_fps_{}.mp4".format(required_fps))

#%%

# creating the directory to save the selected frames

selected_frames_directory = 'selected_frames_60fps'

if not os.path.exists(selected_frames_directory):
    os.makedirs(selected_frames_directory)

#%%

vidcap = cv2.VideoCapture('tennis_video_fps_60.mp4')

# creating the avg hash for the reference court image
reference_hash = imagehash.average_hash(Image.open('court_reference.jpg'))

success,image = vidcap.read()
image2 = BGR_to_RGB(image)

count = 0
frames = []
diffs = []

# looping over the video frame-by-frame
while success:
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file          
    count = count + 1

    # hash0 = imagehash.average_hash(Image.fromarray(first_image))
    hash1 = imagehash.average_hash(Image.fromarray(image2))
    diff = hash1 - reference_hash
    diffs.append(diff)
    
    # if the hamming distance between reference image and query image is <20, then save the image
    if diff<20:
        frames.append(count)
        cv2.imwrite("{}/frame_{}_diff_{}.jpg".format(selected_frames_directory,count,diff), image)
        
    success,image = vidcap.read()
    image2 = BGR_to_RGB(image)  
