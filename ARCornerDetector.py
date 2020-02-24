'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       ARCornerDetector.py
@date       2020/02/20
@brief      Class to detect the corners of AR tags in an image
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2

'''
@brief      AR tag corner detector for a BGR image
'''
class ARCornerDetector:
    image = np.array([])                # Original image
    blurredImage = np.array([])         # Blurred image after filtering
    thresholdedImage = np.array([])     # Thresholded image
    tagContours = np.array([])          # Array of all tag contours
    tagCornerSets = np.array([])        # Array of all tag corner sets


    def __init__(self):
        pass


    '''
    @brief      Finds all AR tag contours in a thresholded image
    @param      thresholdedImage    Thresholded grayscale image
    @return     self.tagContours    NumPy array of detected AR tag contours
    '''
    def getTagContours(self, thresholdedImage):
        # Find all contours in the thresholded image
        contours, hierarchy = \
            cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Extract all contours in level 1 of the contour hierarchy. These are the AR tag contours.
        tagContoursList = []
        for contourInfo in hierarchy[0]:
            # If a contour has no parents, it is in level 0. These are the paper contours.
            # Its child will be the AR tag contour
            if (contourInfo[3] == -1):
                tagContoursList.append(contours[contourInfo[2]])

        # Convert tagContoursList from list to NumPy array
        self.tagContours = np.array(tagContoursList)

        return self.tagContours


    '''
    @brief      Finds all AR tag corner sets in a BGR image
    @param      image                   BGR image
    @return     self.tagCornerSets      NumPy array of detected AR tag corner sets
    '''
    def getTagCorners(self, image):
        self.image = image

        # Apply median blur to remove pixel noise
        self.blurredImage = cv2.medianBlur(self.image, 5)

        # Convert image to grayscale and apply binary thresholding to sharpen black/white areas
        gray = cv2.cvtColor(self.blurredImage, cv2.COLOR_BGR2GRAY)
        retVal, self.thresholdedImage = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Get AR tag contours
        self.getTagContours(self.thresholdedImage)

        # Identify AR tag corners from the tag contours
        tagCornerSetsList = []
        for contour in self.tagContours:
            # Find minimum points to approximate a contour
            epsilon = 1
            cornerSet = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Continue to increase the error tolerance until the approximation produces four points
            while (True):
                if (np.shape(cornerSet)[0] == 4):
                    break

                epsilon = epsilon + 1
                cornerSet = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Reverse the order of the corner coordinates so they are 
            # arranged closckwise around the tag
            cornerSet = cornerSet.tolist()
            cornerSet.reverse()
            cornerSet = np.array(cornerSet)

            # Append the set of corners to a list
            tagCornerSetsList.append(cornerSet)
        
        # Convert tagCornerSetsList from list to NumPy array
        self.tagCornerSets = np.array(tagCornerSetsList)

        return self.tagCornerSets


    '''
    @brief      Demonstration function that can show the images, contours and points
    @param      image                   BGR image
    @return     void
    '''
    def visualization(self, image):
        # Resize image for easy viewing
        import imutils
        image = imutils.resize(image, width=600)

        # Run algorithm to find AR tag corners
        self.getTagCorners(image)

        # Convert thresholded image to BGR and make a copy
        thresholdedBGR = cv2.cvtColor(self.thresholdedImage, cv2.COLOR_GRAY2BGR)
        thresholdCopy = thresholdedBGR.copy()

        # Draw AR tag contours and points
        cv2.drawContours(thresholdedBGR, self.tagContours, -1, (0,0,255), thickness=2)

        for cornerSet in self.tagCornerSets:
            cv2.circle(thresholdCopy, tuple(cornerSet[0].tolist()[0]), 3, color=(255,0,0))
            cv2.circle(thresholdCopy, tuple(cornerSet[1].tolist()[0]), 3, color=(0,255,0))
            cv2.circle(thresholdCopy, tuple(cornerSet[2].tolist()[0]), 3, color=(0,0,255))
            cv2.circle(thresholdCopy, tuple(cornerSet[3].tolist()[0]), 3, color=(255,255,255))

        #cv2.imshow('Original', self.image)
        #cv2.imshow('Median Blurring', self.blurredImage)
        cv2.imshow('Contours', thresholdedBGR)
        cv2.imshow('Corners', thresholdCopy)
        cv2.waitKey(0)



if __name__ == '__main__':
    imageFilename = 'multiple.png'
    image = cv2.imread(imageFilename)
    
    arCornerDetector = ARCornerDetector()
    arCornerDetector.visualization(image)

