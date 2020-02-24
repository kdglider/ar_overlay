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
import ARCornerDetector

'''
@brief      Decoder which determines the orientation, IDs and transformation matrices of all
            AR tands in an image
'''
class ARTagDecoder:
    arCornerDetector = ARCornerDetector()

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
        U, S, VT = svd(A)
        V = np.transpose(VT)
        H = V[:,-1]
        H = np.reshape(H, (3,3))

        return H

    


if __name__ == '__main__':
    imageFilename = 'sample.png'
    image = cv2.imread(imageFilename)
    
    arCornerDetector = ARCornerDetector()
    arCornerDetector.visualization(image)