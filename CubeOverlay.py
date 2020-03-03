'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       CubeOverlay.py
@date       2020/02/20
@brief      Application to overlay a cube on an AR tag in a video stream
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from ARCornerDetector import ARCornerDetector
from ARTagDecoder import ARTagDecoder

'''
@brief      AR application which overlays a cube on an AR tag in a video stream
'''
class CubeOverlay:
    arDetector = ARCornerDetector()     # Corner detector object
    arDecoder = ARTagDecoder()          # Tag decoder object

    thresholdedImage = np.array([])     # Thresholded image frame from arDetector
    tagCornerSets = np.array([])        # Array of all tag corner sets from arDetector
    correctedTagCorners = np.array([])  # Set of tag corners corrected for orientation
    tagContour = np.array([])           # Tag contour
    K = np.array([])                    # Camera intrinsic matrix


    # Initialize constructor with a camera intrinsic matrix
    def __init__(self, K):
        self.K = K


    '''
    @brief      Projects cube points from world to image frame and draws the cube in the video
    @param      cubePoints      8x3 NumPy array of 8 points that defines the cube to overlay
    @param      frame           Current video frame that is being read
    @param      tagCornerSet    NumPy array of the corners of the tag to overlay cube on
    @param      tagContour      Contour of the tag to overlay cube on
    @return     frameCopy       Copy of the video frame with the cube overlaid on AR tag
    '''
    def projectCube(self, cubePoints, frame, tagCornerSet, tagContour):
        # Reshape tagCornerSet to correct shape if it is not already so
        tagCorners = np.reshape(tagCornerSet, (4,2))

        # Define cube base points in 2D world coordinates
        cubeBase = np.array([cubePoints[0][0:2],
                             cubePoints[1][0:2],
                             cubePoints[2][0:2],
                             cubePoints[3][0:2]])

        # Get homography matrix from the cube base to tag corners
        H = self.arDecoder.getHomography(cubeBase, tagCorners)

        # Use the homography and camera intrinsic matrices to construct the projection matrix
        # from the world frame to the image frame
        P = self.arDecoder.getProjectionMatrix(H, self.K)

        # Create empty 8x2 matrix to hold the projected cube points in the image frame
        projCubePoints = np.zeros((np.shape(cubePoints)[0] , 2))

        # Project store each cube point
        for i in range(np.shape(cubePoints)[0]):
            # Construct augmented vector and apply projection
            projPoint = np.matmul(P, np.append(cubePoints[i], 1))

            # Normalize the projected x,y values and reconstruct a 2D pixel vector
            projectedPixel = np.array([projPoint[0]/projPoint[2], \
                                       projPoint[1]/projPoint[2]])

            projCubePoints[i] = projectedPixel
        
        # Round pixel values and convert to integers
        projCubePoints = np.round(projCubePoints)
        projCubePoints = projCubePoints.astype(int)

        # Create a copy of the video frame and draw the projected cube
        frameCopy = frame.copy()

        # Base of cube
        cv2.line(frameCopy, tuple(projCubePoints[0]), tuple(projCubePoints[1]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[1]), tuple(projCubePoints[2]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[2]), tuple(projCubePoints[3]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[3]), tuple(projCubePoints[0]), color=(0,255,0), thickness=2)
        
        # Top of cube
        cv2.line(frameCopy, tuple(projCubePoints[4]), tuple(projCubePoints[5]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[5]), tuple(projCubePoints[6]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[6]), tuple(projCubePoints[7]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[7]), tuple(projCubePoints[4]), color=(0,255,0), thickness=2)
        
        # Sides of cube
        cv2.line(frameCopy, tuple(projCubePoints[0]), tuple(projCubePoints[4]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[1]), tuple(projCubePoints[5]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[2]), tuple(projCubePoints[6]), color=(0,255,0), thickness=2)
        cv2.line(frameCopy, tuple(projCubePoints[3]), tuple(projCubePoints[7]), color=(0,255,0), thickness=2)

        return frameCopy


    '''
    @brief      Applies projectCube() only to the tag in the frame with the desired ID
    @param      cubePoints      8x3 NumPy array of 8 points that defines the cube to overlay
    @param      frame           Current video frame that is being read
    @param      desiredTagID    ID of the desired tag to overlay cube on
    @return     modifiedFrame   Copy of the video frame with the cube overlaid on AR tag
    '''
    def overlayCorrectTag(self, cubePoints, frame, desiredTagID):
        # Get thresholded image from arDetector and all tag corner sets in the current frame
        self.tagCornerSets = self.arDetector.getTagCorners(frame)
        self.thresholdedImage = self.arDetector.thresholdedImage
        
        # Loop through all tags in the frame and check if they have the correct ID
        IDFound = False
        for i in range(np.shape(self.tagCornerSets)[0]):
            # Get current tag in loop
            tagCorners = self.tagCornerSets[i]

            # Decode tag ID and get the corrected corners
            tagID, self.correctedTagCorners = \
                self.arDecoder.decodeTag(self.thresholdedImage, tagCorners)

            # Update tagContour to match that of arDetector
            self.tagContour = self.arDetector.tagContours[i]

            # If the correct tag has been found, exit the loop
            if (tagID == desiredTagID):
                IDFound = True
                break
        
        # If the desired tag has been found in the frame, apply projectCube()
        # Else, do nothing and return the unmodified frame
        if (IDFound == True):
            modifiedFrame = self.projectCube(cubePoints, frame, self.correctedTagCorners, self.tagContour)
            return modifiedFrame
        else:
            return frame

    
    '''
    @brief      Demonstration function to run the cube overlay application
    @param      cubePoints      8x3 NumPy array of 8 points that defines the cube to overlay
    @param      videoFile       Video file to visualize overlay
    @param      desiredTagID    ID of the desired tag to overlay cube on
    @param      saveVideo       Boolean flag to select whether or not to save the modifed video
    @return     void
    '''
    def runApplication(self, cubePoints, videoFile, desiredTagID, saveVideo=False):
        # Create video stream object
        videoCapture = cv2.VideoCapture(videoFile)
        # videoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        
        # Define video codec and output file if video needs to be saved
        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 720p 30fps video
            out = cv2.VideoWriter('CubeOverlayOutput.mp4', fourcc, 30, (720, 480))

        # Continue to process frames if the video stream object is open
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            # Continue processing if a valid frame is received
            if ret == True:
                # Overlay cube on tag
                frame = self.overlayCorrectTag(cubePoints, frame, desiredTagID)

                # Save video if desired, resizing frame to 720p
                if (saveVideo == True):
                    out.write(cv2.resize(frame, (720, 480)))
                
                # Display frame to the screen in a video preview
                cv2.imshow("Frame", cv2.resize(frame, (720, 480)))

                # Exit if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # If the end of the video is reached, wait for final user keypress and exit
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

    # Select ID of the desired tag to overlay cube on
    desiredTagID = 3

    # Choose whether or not to save the output video
    saveVideo = False

    # Run application
    cubeOverlay = CubeOverlay(K)
    cubeOverlay.runApplication(cubePoints, videoFile, desiredTagID, saveVideo)






