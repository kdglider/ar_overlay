'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       ARTagDecoder.py
@date       2020/02/20
@brief      Class with methods to correct for AR tag orientation, determine the ID and 
            determine the projection matrix
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
from ARCornerDetector import ARCornerDetector

'''
@brief      Decoder with methods that correct for the orientation of an AR tag, determine the 
            ID and determine the projection matrix
'''
class ARTagDecoder:
    thresholdedImage = np.array([])     # Thresholded image that contains the tag
    tagCorners = np.array([])           # Set of tag corners
    correctedTagCorners = np.array([])  # Set of tag corners corrected for orientation
    tag = np.array([])                  # 8x8 matrix encoding the 64 cells of an unwarpped AR tag
    tagID = 0                           # Tag integer ID


    def __init__(self):
        pass


    '''
    @brief      Finds the homography matrix between sets with four points each
    @param      srcPoints   4x2 NumPy array of source points in pixel coordinates
    @param      dstPoints   4x2 NumPy array of destination points in pixel coordinates
    @return     H           3x3 NumPy homography matrix
    '''
    def getHomography(self, srcPoints, dstPoints):
        # Ensure incoming point sets are of a 4x2 matrix
        x = np.reshape(srcPoints, (4,2))
        xp = np.reshape(dstPoints, (4,2))

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
    @brief      Finds the projection matrix from a homography matrix and camera intrinsic matrix
    @param      H   3x3 homography matrix between two sets of four corners each
    @param      K   3x3 Camera intrinsic matrix
    @return     P   3x3 NumPy homography matrix
    '''
    def getProjectionMatrix(self, H, K):
        # Get K inverse matrix
        invK = np.linalg.inv(K)

        # Calculate Lambda scale factor
        invKh1 = np.matmul(invK, H[:, 0])
        invKh2 = np.matmul(invK, H[:, 1])
        Lambda = ((np.linalg.norm(invKh1) + np.linalg.norm(invKh2))/2) ** -1

        # Calculate B matrix
        B = np.array([])
        B_Tilda = Lambda * np.matmul(invK, H)
        if (np.linalg.det(B_Tilda) < 0):
            B = -B_Tilda
        else:
            B = B_Tilda

        # Extract translation and rotation vectors
        r1 = np.array(B[:,0])
        r2 = np.array(B[:,1])
        r3 = np.cross(r1,r2)
        t = np.array(B[:,2])

        # Stack row vectors and take the transpose to construct [R|t] matrix
        RtT = np.array([r1, r2, r3, t])
        Rt = np.transpose(RtT)

        # Compute final projection matrix
        P = np.matmul(K, Rt)

        return P


    '''
    @brief      Projects a skewed AR tag from an image into an 8x8 matrix to decode for analysis
    @param      thresholdedImage    Grayscale thresholded image containing tag corners
    @param      tagCorners          Corner set of a tag
    @return     unwarppedTag        8x8 matrix encoding the 64 cells of the unwarpped AR tag
    '''
    def unwarpTag(self, thresholdedImage=None, tagCorners=None):
        # If no parameters are given, set default parameters to class attributes
        if (thresholdedImage.all() == None):
            thresholdedImage = self.thresholdedImage
        if (tagCorners.all() == None):
            tagCorners = self.tagCorners

        # Create a small, temporary blank canvas to unwarp tag onto
        canvas = np.zeros((50, 50))

        # Define 4x2 matrix of canvas corners
        # Corners are ordered in clockwise fashion from the top-left of a tag
        canvasCorners = np.array([[0 , 0] , [np.shape(canvas)[1]-1 , 0], \
            [np.shape(canvas)[1]-1 , np.shape(canvas)[0]-1] , [0 , np.shape(canvas)[0]-1]])

        # Get homography matrix from the canvas corners to the tag corners
        H = self.getHomography(canvasCorners, tagCorners)

        # Define the pixel limits of the thresholded image
        imageXLimit = np.shape(thresholdedImage)[1]-1
        imageYLimit = np.shape(thresholdedImage)[0]-1

        # Fill all pixels of the canvas with the correct grayscale value from the thresholded image tag
        # Note that the OpenCV image coordinate system is the reverse of the indexing used to access
        # NumPy array values, hence the x,y reversal when changing the grayscale values
        for x in range(np.shape(canvas)[1]):
            for y in range(np.shape(canvas)[0]):
                # Construct augmented vector and apply homography transformation
                projectedVector = np.matmul(H, np.array([x,y,1]))

                # Normalize the projected vector x,y values and reconstruct a 2D pixel vector
                projectedPixel = np.array([projectedVector[0]/projectedVector[2], \
                                           projectedVector[1]/projectedVector[2]])
                
                # Round pixel values and convert to integers for indexing
                projectedPixel = np.round(projectedPixel)
                projectedPixel = projectedPixel.astype(int)

                # Ensure pixel coordinate are bounded to the image limits
                projectedPixel[0] = min(imageXLimit, max(0, projectedPixel[0]))
                projectedPixel[1] = min(imageYLimit, max(0, projectedPixel[1]))

                # Change the respective canvas grayscale value to match that of the thresholded image
                canvas[y,x] = thresholdedImage[projectedPixel[1], projectedPixel[0]]
        
        # Resize canvas to the required 8x8 matrix
        unwarppedTag = cv2.resize(canvas, (8,8))
        self.tag = unwarppedTag

        return unwarppedTag


    '''
    @brief      Unwarps a tag from an image, corrects the orientation and decodes the ID
    @param      thresholdedImage    Grayscale thresholded image from which the tag corners were taken
    @param      tagCorners          Corner set of a tag
    @return     tagID               8x8 matrix encoding the 64 cells of the unwarpped AR tag
    @return     correctedTagCorners Tag corner set corrected for orientation
    '''
    def decodeTag(self, thresholdedImage, tagCorners):
        # Update class attributes with parameters
        self.thresholdedImage = thresholdedImage
        self.tagCorners = np.reshape(tagCorners, (4,2))

        # Unwarp image tag to generate 8x8 matrix to decode orientation and ID
        tag = self.unwarpTag(thresholdedImage, tagCorners)

        # Correct tag orientation and record corrected corners 
        # With the tag upright, the top-left corner should be the first row in the array, with 
        # the remaining corners added in CW order

        # If orientation square is in the top-left, rotate tag and corners 180 degrees
        if (tag[2,2] == 255):
            tag = np.rot90(tag, k=2)
            self.correctedTagCorners = np.roll(self.tagCorners, 2, axis=0)

        # If orientation square is in the bottom-left, rotate tag and corners 90 degrees CCW
        elif (tag[5,2] == 255):
            tag = np.rot90(tag, k=1, axes=(0,1))
            self.correctedTagCorners = np.roll(self.tagCorners, -1, axis=0)
        
        # If orientation square is in the top-right, rotate tag and corners 90 degrees CW
        elif (tag[5,2] == 255):
            tag = np.rot90(tag, k=1, axes=(1,0))
            self.correctedTagCorners = np.roll(self.tagCorners, 1, axis=0)

        # If orientation square is in the bottom-right; do nothing with the tag or corners
        else:
            self.correctedTagCorners = self.tagCorners

        self.tag = tag

        # With tag in the correct orientation, construct ID code in binary (python string format)
        # from the middle four squares (top-left LSB and moving CW to MSB)
        binaryID = str(int(tag[4,3]==255)) + str(int(tag[4,4]==255)) + \
                    str(int(tag[3,4]==255)) + str(int(tag[3,3]==255))
        
        # Convert binary ID to an integer
        self.tagID = int(binaryID, 2)

        return self.tagID, self.correctedTagCorners



if __name__ == '__main__':
    # Sample image
    imageFilename = 'sample_images/multipleTags.png'
    image = cv2.imread(imageFilename)

    # Sample camera intrinsic matrix
    K = np.array([[1406.08415449821,    2.206797873085990,      1014.136434174160],
                  [0,                   1417.99930662800,       566.347754321696],
                  [0,                   0,                      1]])
    
    # Create detector and decoder instances
    arCornerDetector = ARCornerDetector()
    arDecoder = ARTagDecoder()

    # Detect corners
    arCornerDetector.getTagCorners(image)

    # Sample canvas
    canvasCorners = np.array([[0 , 0] , [100, 0], \
                              [100 , 100] , [0 , 100]])

    # Generae projection matrix with first cornerset in arCornerDetector
    H = arDecoder.getHomography(canvasCorners, np.reshape(arCornerDetector.tagCornerSets[0], (4,2)))
    P = arDecoder.getProjectionMatrix(H, K)
    print('The projection matrix is: ')
    print(P)

    
    # Print uncorrected tag corners, corrected tag corners and tag ID
    arDecoder.decodeTag(arCornerDetector.thresholdedImage, arCornerDetector.tagCornerSets[0])

    print('The uncorrected tag corners are: ')
    print(arCornerDetector.tagCornerSets[0])

    print('The corrected tag corners are: ')
    print(arDecoder.correctedTagCorners)

    print('The tag ID is: ')
    print(arDecoder.tagID)

    # cv2.imshow("Tag", cv2.resize(arDecoder.tag, (500,500)))
    # cv2.waitKey(0)
