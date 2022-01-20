#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:12:38 2021

@author: keshav
"""

import cv2
from PIL import Image
import imagehash
import numpy as np
# import opencv_wrapper as cvw

#%%

from court_detector import CourtDetector
court_detector = CourtDetector()

#%%
# construct the argument parse and parse the arguments
# load the image
image = cv2.imread("court_reference.jpg")
# image = cv2.imread("test_reference_similarity.jpg")

#%%

# detect all the vertical and horizontal lines of the tennis court

lines = court_detector.detect(image)

#%%

# line 1 and 2 are enough to represent the outermost point of the tennis court. 
# So we are capturing only start and end points of line 1 and 2

line_count = 0
points = []
for i in range(0, len(lines), 4):
    
    line_count += 1
    x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]

    if line_count==1:
        points.append(tuple([int(x1),int(y1)]))
        points.append(tuple([int(x2),int(y2)]))
    elif line_count==2:
        points.append(tuple([int(x1),int(y1)]))
        points.append(tuple([int(x2),int(y2)]))        
    # points.append()
    # points.append(tuple([int(x2),int(y2)]))
    
court_points_sorted = sorted(set(points))

#%%

# Saving the court extreme points for future reference

import pickle
with open('court_points_sorted.pkl','wb') as f:
    pickle.dump(court_points_sorted,f)
    