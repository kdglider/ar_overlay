'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       TBD
@date       2020/02/20
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
import imutils

image = cv2.imread('multiple.PNG')
image = imutils.resize(image, width=600)

# g = cv2.GaussianBlur(image, (7,7), 0)

# Apply median blur to remove pixel noise
blurredImage = cv2.medianBlur(image, 5)

# Convert image to grayscale and apply binary thresholding to sharpen black/white areas
gray = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)
retVal, thresholdedImage = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

contours, hierarchy = \
    cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Extract all contours in level 1 of the contour hierarchy. These are the AR tag contours
tagContours = []
for contourInfo in hierarchy[0]:
    # If a contour has no parents, it is in level 0. These are the paper contours.
    # Its child will be the AR tag contour
    if (contourInfo[3] == -1):
        tagContours.append(contours[contourInfo[2]])

tagContours = np.array(tagContours)

copy = cv2.cvtColor(thresholdedImage.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(copy, tagContours, -1, (0,0,255), thickness=2)

#cv2.imshow('image', image)
#cv2.imshow('Median', blurredImage)
cv2.imshow('threshold', thresholdedImage)
cv2.imshow('contour', copy)
cv2.waitKey(0)
