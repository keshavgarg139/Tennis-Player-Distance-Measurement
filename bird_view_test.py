#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:09:29 2021

@author: keshav
"""

"""
Test file to observe the working of imutils.perspective.four_point_transform
"""

from imutils import perspective
import numpy as np
import cv2

#%%

import pickle
with open('court_points_sorted.pkl','rb') as f:
    court_points_sorted = pickle.load(f)

#%%%%%

# court_points_sorted_y_reversed = []
# for p in court_points_sorted:
#     p2 = [p[0],720-p[1]]
#     court_points_sorted_y_reversed.append(p2)
    
#%%%%%

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
# load the notecard code image, clone it, and initialize the 4 points
# that correspond to the 4 corners of the notecard

#%%

extended_court_points2 = [extended_court_points[i] for i in [1,2,3,0]]

#%%

img = cv2.imread("court_reference.jpg")
clone = img.copy()

#%%

# pts = np.array([(73, 239), (356, 117), (475, 265), (187, 443)])
pts = np.array(extended_court_points2)

# loop over the points and draw them on the cloned image
for (x, y) in pts:
    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

# apply the four point tranform to obtain a "birds eye view" of
# the notecard
warped = perspective.four_point_transform(img, pts)

#%%

cv2.imwrite('warped.jpg',warped)

#%%

#%%

img = cv2.imread("warped.jpg")
clone = img.copy()

pts = np.array([(0, 620), (243, 1), (908, 1), (1070, 617)])
# pts = np.array(extended_court_points2)

# loop over the points and draw them on the cloned image
for (x, y) in pts:
    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

# apply the four point tranform to obtain a "birds eye view" of
# the notecard
warped = perspective.four_point_transform(img, pts)

cv2.imwrite('warped2.jpg',warped)

#%%

# show the original and warped images
cv2.imshow("Original", clone)
cv2.imshow("Warped", warped)
cv2.waitKey(0)