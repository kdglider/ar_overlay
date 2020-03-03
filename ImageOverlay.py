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

'''
@brief      AR application which overlays an image on an AR tag in a video stream
'''
class ImageOverlay:
    arDetector = ARCornerDetector()     # Corner detector object
    arDecoder = ARTagDecoder()          # Tag decoder object

    thresholdedImage = np.array([])     # Thresholded image frame from arDetector
    tagCornerSets = np.array([])        # Array of all tag corner sets from arDetector
    correctedTagCorners = np.array([])  # Set of tag corners corrected for orientation
    tagContour = np.array([])           # Tag contour


    def __init__(self):
        pass

    '''
    @brief      Replaces the tag BGR values in a frame with the BGR values of an image
    @param      image           BGR image to overlay on the AR tag
    @param      frame           Current video frame that is being read
    @param      tagCornerSet    NumPy array of the corners of the tag to overlay image on
    @param      tagContour      Contour of the tag to overlay image on
    @return     frameCopy       Copy of the video frame with the image overlaid on AR tag
    '''
    def warpImage(self, image, frame, tagCornerSet, tagContour):
        # Reshape tagCornerSet to correct shape if it is not already so
        tagCorners = np.reshape(tagCornerSet, (4,2))

        # Identify the image index limits and corner pixel coordinates
        imageXLimit = np.shape(image)[1]-1
        imageYLimit = np.shape(image)[0]-1
        imageCorners = np.array([[0 , 0] , [imageXLimit , 0], \
                                 [imageXLimit , imageYLimit] , [0 , imageYLimit]])

        # Get homography matrix from tag corners to image corners
        H = self.arDecoder.getHomography(tagCorners, imageCorners)
        
        # Define x/y pixel limits of the minimum bounding box in the frame that contains the tag
        xLow = min(tagCorners[:,0])
        xHigh = max(tagCorners[:,0])
        yLow = min(tagCorners[:,1])
        yHigh = max(tagCorners[:,1])

        # Make a copy of the frame to change BGR values
        frameCopy = frame.copy()

        # Loop through bounding box and replace tag BGR values with BGR values from the image
        # Search 1 pixel out of the bounding box on all sides for safety
        for x in range(xLow-1 , xHigh+2):
            for y in range(yLow-1 , yHigh+2):
                # Change pixel BGR value only if it lies within the tag's contour
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

                    # Change the respective frame BGR values to match those of the image
                    frameCopy[y,x,0] = image[projectedPixel[1], projectedPixel[0], 0]
                    frameCopy[y,x,1] = image[projectedPixel[1], projectedPixel[0], 1]
                    frameCopy[y,x,2] = image[projectedPixel[1], projectedPixel[0], 2]
        
        return frameCopy


    '''
    @brief      Applies warpImage() only to the tag in the frame with the desired ID
    @param      image           BGR image to overlay on the AR tag
    @param      frame           Current video frame that is being read
    @param      desiredTagID    ID of the desired tag to overlay image on
    @return     modifiedFrame   Copy of the video frame with the image overlaid on AR tag
    '''
    def overlayCorrectTag(self, image, frame, desiredTagID):
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
        
        # If the desired tag has been found in the frame, apply warpImage()
        # Else, do nothing and return the unmodified frame
        if (IDFound == True):
            modifiedFrame = self.warpImage(image, frame, self.correctedTagCorners, self.tagContour)
            return modifiedFrame
        else:
            return frame

    
    '''
    @brief      Demonstration function to run the image overlay application
    @param      imageFile       Image file of BGR image to overlay on an AR tag
    @param      videoFile       Video file to visualize overlay
    @param      desiredTagID    ID of the desired tag to overlay image on
    @param      saveVideo       Boolean flag to select whether or not to save the modifed video
    @return     void
    '''
    def runApplication(self, imageFile, videoFile, desiredTagID, saveVideo=False):
        # Read image and create video stream object
        image = cv2.imread(imageFile)
        videoCapture = cv2.VideoCapture(videoFile)
        # videoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        # Define video codec and output file if video needs to be saved
        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 720p 30fps video
            out = cv2.VideoWriter('ImageOverlayOutput.mp4', fourcc, 30, (720, 480))

        # Continue to process frames if the video stream object is open
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            # Continue processing if a valid frame is received
            if ret == True:
                # Overlay image on tag
                frame = self.overlayCorrectTag(image, frame, desiredTagID)

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
    # Select file of image to be superimposed
    # Select video file with AR tag(s) 
    imageFile = 'sample_images/Lena.png'
    videoFile = 'sample_videos/multipleTags.mp4'

    # Select ID of the desired tag to overlay cube on
    desiredTagID = 3

    # Choose whether or not to save the output video
    saveVideo = False

    # Create object instance and run application
    imageOverlay = ImageOverlay()
    imageOverlay.runApplication(imageFile, videoFile, desiredTagID, saveVideo)





