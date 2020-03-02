'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       ImageOverlay.py
@date       2020/02/20
@brief      Application to overlay an image on an AR tag in a video stream
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from ARCornerDetector import ARCornerDetector
from ARTagDecoder import ARTagDecoder


class ImageOverlay:
    arDetector = ARCornerDetector()     
    arDecoder = ARTagDecoder()

    thresholdedImage = np.array([])     # Thresholded image frame from arDetector
    tagCornerSets = np.array([])        # Array of all tag corner sets from arDetector
    correctedTagCorners = np.array([])  # Set of tag corners corrected for orientation
    tagContour = np.array([])           # Tag contour


    def __init__(self):
        pass


    def warpImage(self, image, frame, tagCornerSet, tagContour):
        tagCorners = np.reshape(tagCornerSet, (4,2))

        imageXLimit = np.shape(image)[1]-1
        imageYLimit = np.shape(image)[0]-1
        imageCorners = np.array([[0 , 0] , [imageXLimit , 0], \
                                 [imageXLimit , imageYLimit] , [0 , imageYLimit]])

        H = self.arDecoder.getHomography(tagCorners, imageCorners)
        
        xLow = min(tagCorners[:,0])
        xHigh = max(tagCorners[:,0])
        yLow = min(tagCorners[:,1])
        yHigh = max(tagCorners[:,1])

        frameCopy = frame.copy()
        for x in range(xLow-1 , xHigh+2):
            for y in range(yLow-1 , yHigh+2):
                if (cv2.pointPolygonTest(tagContour, (x,y), measureDist=False) == True):
                    # Construct augmented vector and apply homography transformation
                    projectedVector = np.matmul(H, np.array([x,y,1]))

                    # Normalize the project x,y values and reconstruct a 2D pixel vector
                    projectedPixel = np.array([projectedVector[0]/projectedVector[2], \
                                            projectedVector[1]/projectedVector[2]])
                    
                    # Ensure pixel coordinate are bounded to the image limits
                    projectedPixel[0] = min(imageXLimit, max(0, projectedPixel[0]))
                    projectedPixel[1] = min(imageYLimit, max(0, projectedPixel[1]))
                    
                    # Round pixel values and convert to integers for indexing
                    projectedPixel = np.round(projectedPixel)
                    projectedPixel = projectedPixel.astype(int)

                    # Change the respective frame BGR values to match that of the image
                    frameCopy[y,x,0] = image[projectedPixel[1], projectedPixel[0], 0]
                    frameCopy[y,x,1] = image[projectedPixel[1], projectedPixel[0], 1]
                    frameCopy[y,x,2] = image[projectedPixel[1], projectedPixel[0], 2]
        
        return frameCopy


    def overlayCorrectTag(self, image, frame, desiredTagID):
        self.tagCornerSets = self.arDetector.getTagCorners(frame)
        self.thresholdedImage = self.arDetector.thresholdedImage
        
        IDFound = False
        for i in range(np.shape(self.tagCornerSets)[0]):
            tagCorners = self.tagCornerSets[i]
            tagID, self.correctedTagCorners = \
                self.arDecoder.decodeTag(self.thresholdedImage, tagCorners)
            self.tagContour = self.arDetector.tagContours[i]

            if (tagID == desiredTagID):
                IDFound = True
                break
        
        if (IDFound == True):
            modifiedFrame = self.warpImage(image, frame, self.correctedTagCorners, self.tagContour)
            return modifiedFrame
        else:
            return frame

    
    def runApplication(self, imageFile, videoFile, desiredTagID, saveVideo=False):
        image = cv2.imread(imageFile)
        videoCapture = cv2.VideoCapture(videoFile)
        # videoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('ImageOverlayOutput.mp4', fourcc, 30, (720, 480))

        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            if ret == True:
                frame = self.overlayCorrectTag(image, frame, desiredTagID)

                if (saveVideo == True):
                    out.write(cv2.resize(frame, (720, 480)))
                
                cv2.imshow("Frame", cv2.resize(frame, (720, 480)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                cv2.waitKey(0)
                break
        
        # Release video and file object handles
        videoCapture.release()
        if (saveVideo == True):
            out.release()
        
        print('Video and file handles closed')
        


if __name__ == '__main__':
    # Select file of image to be superimposed
    # Select video file with AR tag(s) 
    imageFile = 'sample_images/Lena.png'
    videoFile = 'sample_videos/multipleTags.mp4'

    # Select ID of the desired tag to overlay cube on
    desiredTagID = 3

    # Choose whether or not to save the output video
    saveVideo = True

    imageOverlay = ImageOverlay()
    imageOverlay.runApplication(imageFile, videoFile, desiredTagID, saveVideo)





