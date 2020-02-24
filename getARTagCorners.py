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

#g = cv2.GaussianBlur(image, (7,7), 0)

# Apply median blur to remove pixel noise
blurredImage = cv2.medianBlur(image, 5)

# Convert image to grayscale and apply binary thresholding to sharpen black/white areas
gray = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)
retVal, thresholdedImage = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Find all contours in the thresholded image
contours, hierarchy = \
    cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

thresholdedImage = cv2.cvtColor(thresholdedImage, cv2.COLOR_GRAY2BGR)
thresholdCopy = thresholdedImage.copy()

# Extract all contours in level 1 of the contour hierarchy. These are the AR tag contours
tagContours = []
for contourInfo in hierarchy[0]:
    # If a contour has no parents, it is in level 0. These are the paper contours.
    # Its child will be the AR tag contour
    if (contourInfo[3] == -1):
        tagContours.append(contours[contourInfo[2]])

# Convert tagContours from list to NumPy array
tagContours = np.array(tagContours)

# Identify AR tag corners from the tag contours
ARCornerSets = []
for contour in tagContours:

    # Find minimum points to approximate a contour
    epsilon = 1
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Continue to increase the error tolerance until the approximation produces four points
    while (True):
        if (np.shape(approx)[0] == 4):
            break

        epsilon = epsilon + 1
        approx = cv2.approxPolyDP(contour, epsilon, True)

    # Reverse the order of the corner coordinates so they are arranged closckwise around the tag
    approx = approx.tolist()
    approx.reverse()
    approx = np.array(approx)

    cv2.circle(thresholdCopy, tuple(approx[0].tolist()[0]), 3, color=(255,0,0))
    cv2.circle(thresholdCopy, tuple(approx[1].tolist()[0]), 3, color=(0,255,0))
    cv2.circle(thresholdCopy, tuple(approx[2].tolist()[0]), 3, color=(0,0,255))
    cv2.circle(thresholdCopy, tuple(approx[3].tolist()[0]), 3, color=(255,255,255))

    # Append the set of corners to a list
    ARCornerSets.append(approx)

cv2.drawContours(thresholdedImage, tagContours, -1, (0,0,255), thickness=2)

#cv2.imshow('Original', image)
#cv2.imshow('Median Blurring', blurredImage)
cv2.imshow('Contours', thresholdedImage)
cv2.imshow('Corners', thresholdCopy)
cv2.waitKey(0)
