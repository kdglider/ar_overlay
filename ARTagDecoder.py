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

        # Compute the SVD for A. The last vector of V will be the solution to the H matrix
        U, s, VT = np.linalg.svd(A)
        V = np.transpose(VT)
        H = V[:,-1]
        #print(s)
        #print(V)
        #print(H)
        # A_new = U @ np.diag(s) @ VT
        # print(A - A_new)
        H = H / H[8]
        H = np.reshape(H, (3,3))
        #print(H)

        return H

    
    '''
    @brief      Projects a skewed AR tag into an unskewed 500x500 image for analysis
    @param      thresholdedImage    Thresholded image from which the tag contour was taken
    @param      tagContour          Contour of a tag
    @param      tagCornerSet        Corner set of a tag
    @return     unskewedTag         500x500 pixel unskewed AR tag
    '''
    def getUnskewedTag(self, thresholdedImage, tagContour, tagCornerSet):
        unskewedTag = np.zeros((500, 500))

        unskewedTagCorners = np.array([[0,0], [0, np.shape(unskewedTag)[0]-1], \
            [np.shape(unskewedTag)[1]-1, np.shape(unskewedTag)[0]-1], [np.shape(unskewedTag)[1]-1, 0]])

        H = self.getHomography(unskewedTagCorners, np.reshape(tagCornerSet, (4,2)))
        #H = self.getHomography(unskewedTagCorners, unskewedTagCorners)
        '''
        print(H)

        print(np.matmul(H, np.array([0,0,1])))
        print(np.matmul(H, np.array([0,100,1])))
        print(np.matmul(H, np.array([100,0,1])))
        print(np.matmul(H, np.array([100,100,1])))'''

        print(np.shape(thresholdedImage))

        print (tagCornerSet)
        projectedVector = np.matmul(H, np.array([0,499,1]))
        projectedPixel = np.array([projectedVector[0]/projectedVector[2], \
                                    projectedVector[1]/projectedVector[2]])
        
        projectedPixel = np.round(projectedPixel)
        projectedPixel = projectedPixel.astype(int)
        print(projectedPixel)
        print(thresholdedImage)

        
        for x in range(np.shape(unskewedTag)[0]):
            for y in range(np.shape(unskewedTag)[1]):
                
                projectedVector = np.matmul(H, np.array([x,y,1]))
                projectedPixel = np.array([projectedVector[0]/projectedVector[2], \
                                           projectedVector[1]/projectedVector[2]])
                
                projectedPixel = np.round(projectedPixel)
                projectedPixel = projectedPixel.astype(int)
                #print(x)
                #print(y)
                #print(projectedPixel)
                unskewedTag[y,x] = thresholdedImage[projectedPixel[1], projectedPixel[0]]
        
        return unskewedTag

if __name__ == '__main__':
    imageFilename = 'sample.png'
    image = cv2.imread(imageFilename)
    
    arCornerDetector = ARCornerDetector()
    arCornerDetector.visualization(image)

    arDecoder = ARTagDecoder()
    tag = arDecoder.getUnskewedTag(arCornerDetector.thresholdedImage, arCornerDetector.tagContours[0], arCornerDetector.tagCornerSets[0])
    cv2.imshow('Tag', tag)
    cv2.waitKey(0)