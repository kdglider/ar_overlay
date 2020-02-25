'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       ARTagDecoder.py
@date       2020/02/20
@brief      Class to determine AR tag orientations, IDs and 
            transformation matrices with respect to the camera
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from ARCornerDetector import ARCornerDetector

'''
@brief      Decoder which determines the orientation, IDs and transformation matrices of all
            AR tands in an image
'''
class ARTagDecoder:
    

    def __init__(self):
        pass


    '''
    @brief      Finds the homography matrix between two sets with four points each
    @param      srcPoints   4x2 NumPy array of source points in pixel coordinates
    @param      dstPoints   4x2 NumPy array of destination points in pixel coordinates
    @return     H           3x3 NumPy homography matrix
    '''
    def getHomography(self, srcPoints, dstPoints):
        x = srcPoints
        xp = dstPoints

        # Generate 8x9 A matrix to solve
        A=np.array([[-x[0,0],-x[0,1],-1,0,0,0,x[0,0]*xp[0,0],x[0,1]*xp[0,0],xp[0,0]],
                    [0,0,0,-x[0,0],-x[0,1],-1,x[0,0]*xp[0,1],x[0,1]*xp[0,1],xp[0,1]],
                    [-x[1,0],-x[1,1],-1,0,0,0,x[1,0]*xp[1,0],x[1,1]*xp[1,0],xp[1,0]],
                    [0,0,0,-x[1,0],-x[1,1],-1,x[1,0]*xp[1,1],x[1,1]*xp[1,1],xp[1,1]],
                    [-x[2,0],-x[2,1],-1,0,0,0,x[2,0]*xp[2,0],x[2,1]*xp[2,0],xp[2,0]],
                    [0,0,0,-x[2,0],-x[2,1],-1,x[2,0]*xp[2,1],x[2,1]*xp[2,1],xp[2,1]],
                    [-x[3,0],-x[3,1],-1,0,0,0,x[3,0]*xp[3,0],x[3,1]*xp[3,0],xp[3,0]],
                    [0,0,0,-x[3,0],-x[3,1],-1,x[3,0]*xp[3,1],x[3,1]*xp[3,1],xp[3,1]]])

        # Compute the SVD for A. The last vector of V will be the solution to the H matrix.
        U, s, VT = np.linalg.svd(A)
        V = np.transpose(VT)
        H = V[:,-1]

        # Normalize H vector using the last element and reshape into a 3x3 matrix
        H = H / H[8]
        H = np.reshape(H, (3,3))

        return H

    
    '''
    @brief      Projects a skewed AR tag into an 8x8 matrix to decode for analysis
    @param      thresholdedImage    Grayscale thresholded image from which the tag contour was taken
    @param      tagCornerSet        Corner set of a tag
    @return     unwarppedTag        8x8 matrix encoding the 64 cells of the unwarpped AR tag
    '''
    def unwarpTag(self, thresholdedImage, tagCornerSet):
        # Create a small, temporary blank canvas to unwarp tag onto
        canvas = np.zeros((50, 50))

        # Define 4x2 matrix of canvas corners
        # Corners are ordered in clockwise fashion from the top-left of a tag
        canvasCorners = np.array([[0 , 0] , [np.shape(canvas)[0]-1 , 0], \
            [np.shape(canvas)[0]-1 , np.shape(canvas)[1]-1] , [0 , np.shape(canvas)[1]-1]])

        # Reshape tagCornerSet into a 4x2 matrix and get homography matrix with the canvas corners
        H = self.getHomography(canvasCorners, np.reshape(tagCornerSet, (4,2)))

        # Fill all pixels of the canvas with the correct grayscale value from the thresholded image tag
        # Note that the OpenCV image coordinate system is the reverse of the indexing used to access
        # NumPy array values, hence the x,y reversal when changing the grayscale values
        for x in range(np.shape(canvas)[1]):
            for y in range(np.shape(canvas)[0]):
                # Construct augmented vector and apply homography transformation
                projectedVector = np.matmul(H, np.array([x,y,1]))

                # Normalize the project x,y values and reconstruct a 2D pixel vector
                projectedPixel = np.array([projectedVector[0]/projectedVector[2], \
                                           projectedVector[1]/projectedVector[2]])
                
                # Round pixel values and convert to integers for indexing
                projectedPixel = np.round(projectedPixel)
                projectedPixel = projectedPixel.astype(int)

                # Change the respective canvas grayscale value to match that of the thresholded image
                canvas[y,x] = thresholdedImage[projectedPixel[1], projectedPixel[0]]
        
        # Resize canvas to the required 8x8 matrix
        unwarppedTag = cv2.resize(canvas, (8,8))
        print(unwarppedTag)
        return unwarppedTag



if __name__ == '__main__':
    imageFilename = 'sample.png'
    image = cv2.imread(imageFilename)
    
    arCornerDetector = ARCornerDetector()
    arCornerDetector.visualization(image)

    arDecoder = ARTagDecoder()
    tag = arDecoder.unwarpTag(arCornerDetector.thresholdedImage, arCornerDetector.tagCornerSets[0])

    tag = cv2.resize(tag, (500,500))
    cv2.imshow("Tag", tag)
    cv2.waitKey(0)
