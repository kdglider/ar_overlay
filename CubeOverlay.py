'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       CubeOverlay.py
@date       2020/02/20
@brief      
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from ARCornerDetector import ARCornerDetector
from ARTagDecoder import ARTagDecoder


class CubeOverlay:
    arDetector = ARCornerDetector()     
    arDecoder = ARTagDecoder()

    thresholdedImage = np.array([])     # Thresholded image frame from arDetector
    tagCornerSets = np.array([])        # Array of all tag corner sets from arDetector
    correctedTagCorners = np.array([])  # Set of tag corners corrected for orientation
    tagContour = np.array([])           # Tag contour
    K = np.array([])


    def __init__(self, K):
        self.K = K

    def projectCube(self, cubePoints, frame, tagCornerSet, tagContour):
        tagCorners = np.reshape(tagCornerSet, (4,2))

        cubeBase = np.array([cubePoints[0][0:2],
                             cubePoints[1][0:2],
                             cubePoints[2][0:2],
                             cubePoints[3][0:2]])

        H = self.arDecoder.getHomography(cubeBase, tagCorners)

        P = self.arDecoder.getProjectionMatrix(H, self.K)

        frameCopy = frame.copy()
        
        projCubePoints = np.zeros((np.shape(cubePoints)[0] , 2))
        for i in range(np.shape(cubePoints)[0]):
            projPoint = np.matmul(P, np.append(cubePoints[i], 1))
            # Normalize the project x,y values and reconstruct a 2D pixel vector
            projectedPixel = np.array([projPoint[0]/projPoint[2], \
                                       projPoint[1]/projPoint[2]])

            projCubePoints[i] = projectedPixel
        
        # Round pixel values and convert to integers
        projCubePoints = np.round(projCubePoints)
        projCubePoints = projCubePoints.astype(int)

        cv2.line(frameCopy, tuple(projCubePoints[0]), tuple(projCubePoints[1]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[1]), tuple(projCubePoints[2]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[2]), tuple(projCubePoints[3]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[3]), tuple(projCubePoints[0]), color=(0,255,0), thickness=2)
        
        cv2.line(frameCopy, tuple(projCubePoints[4]), tuple(projCubePoints[5]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[5]), tuple(projCubePoints[6]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[6]), tuple(projCubePoints[7]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[7]), tuple(projCubePoints[4]), color=(0,255,0), thickness=2)
        
        cv2.line(frameCopy, tuple(projCubePoints[0]), tuple(projCubePoints[4]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[1]), tuple(projCubePoints[5]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[2]), tuple(projCubePoints[6]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[3]), tuple(projCubePoints[7]), color=(0,255,0), thickness=2)

        return frameCopy


    def overlayCorrectTag(self, cubePoints, frame, desiredTagID):
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
            modifiedFrame = self.projectCube(cubePoints, frame, \
                                             self.correctedTagCorners, self.tagContour)
            return modifiedFrame
        else:
            return frame

    
    def runApplication(self, cubePoints, videoFile, desiredTagID, saveVideo=False):
        videoCapture = cv2.VideoCapture(videoFile)

        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('CubeOverlayOutput.mp4', fourcc, 30, (720, 480))

        while(True):
            ret, frame = videoCapture.read()

            if ret == True:
                frame = self.overlayCorrectTag(cubePoints, frame, desiredTagID)

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
    # Define cube vertices, with the z axis pointing down so that the x/y axes match OpenCV convention
    # Ensure that the base points are the first 4, in CW order
    cubePoints = np.array([[0,0,0], 
                           [1,0,0],
                           [1,1,0],
                           [0,1,0],
                           [0,0,-1], 
                           [1,0,-1],
                           [1,1,-1],
                           [0,1,-1]])
    
    # Camera intrinsic matrix
    K = np.array([[1406.08415449821,    2.206797873085990,      1014.136434174160],
                  [0,                   1417.99930662800,       566.347754321696],
                  [0,                   0,                      1]])

    # Select video file and ID of the desired tag to overlay cube on
    videoFile = 'sample_videos/multipleTags.mp4'
    desiredTagID = 3
    saveVideo = False

    # Run application
    cubeOverlay = CubeOverlay(K)
    cubeOverlay.runApplication(cubePoints, videoFile, desiredTagID, saveVideo)






