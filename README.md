# AR Video Overlay Application

## Overview
This project is an augmented reality application where specifc AR tags are recognized in a video stream and overlaid with either an image or a 3D cube. A sample tag is shown below:

![Sample AR Tag with Grid](/Results/AR_grid.png)

The AR tags use an encoding scheme that can be described as follows:

1) The tag can be decomposed into an 8x8 grid of squares, which includes a padding of 2 squares (outer black cells in the image) along the borders. This allows easy detection of the tag when placed on any contrasting background.

2) The inner 4x4 grid (i.e. after removing the padding) has the orientation depicted by a white square in the lower-right corner. This represents the upright position of the tag.

3) Lastly, the inner-most 2x2 grid (i.e. after removing the padding and the orientation grids) encodes the binary representation of the tag’s ID, which is ordered in the clockwise direction from the least significant bit to the most significant bit. So, the top-left square is the least significant bit, and the bottom-left square is
the most significant bit.

The application pipeline developed involves the following steps:

1) AR Tag Detection

Thresholding is performed on the video frame to isolate the black and white regions. Afterwards, contours of the black/white regions are identified and the tag contours are taken to be the ones at level 1 of the hierarchy (level 0 corresponds to that of the paper the tag is printed on). Tag corners can then be extracted from the contours.

2) AR Tag Decoding

The warped tag in the video frame is unwarped into an 8x8 array using a homography transformation. Using the orientation square, the corners of the tag are corrected and the tag ID read from the center 4 squares.

3) Image/Cube Overlay

For image overlay, if the tag with the desired ID is detected in a frame, a homography is performed on the pixels of an image to overlay it onto the tag.

For cube overlay, if the tag with the desired ID is detected in a frame, the projection matrix from the world frame to the image frame is calculated using a homography matrix and a camera intrinsic matrix. The projection matrix is then used to transform all corners of the the cube in 3D space onto the video frame and draws the necessary lines to connect those corners.


## Personnel
Hao Da (Kevin) Dong

Anshuman Singh


## License
This project is licensed under the BSD 3-Clause. Please see LICENSE for additional details and disclaimer.


## Dependencies
The system must have Python 3, NumPy, OpenCV and ImUtils installed. Our systems are Windows-based with Python/NumPy installed as part of [Anaconda](https://www.anaconda.com/distribution/), and with OpenCV 4.1.2. Use the following commands to install the dependencies after installing Anaconda:
```
pip install numpy
pip install opencv-python
pip install imutils
```
Though untested, using a Linux system and/or using packages of a slightly different version should also work. 


## Run Demonstration
Open a terminal and clone our repository into a local directory using:
```
git clone https://github.com/kdglider/ar_overlay.git
```
Change directory to the repository.

To run the image overlay application (Part 2a in the assignment), use:
```
python3 ImageOverlay.py
```
The video of the sample Lena image overlaid on the desired AR tag in the correct orientation will be shown. Certain options can be changed at the bottom of the script, including the desired tag ID, video file, image file and whether or not to save the video.


To run the cube overlay application (Part 2b), use:
```
python3 CubeOverlay.py
```
As with the previous application, options can be changed at the bottom of the script.

The ARCornerDetector and ARTagDecoder classes, both of which are used in the two applications, can also be run independently using:
```
python3 ARCornerDetector.py
python3 ARTagDecoder.py
```
This gives a demonstration of the functionalities of these classes. In particular, running the ARTagDecoder script will give the projection matrix, tag corners (corrected for orientation) and tag ID in a sample image (Part 1).


## Results and Known Issues
Sample screenshots taken from the image and cube overlay videos are shown below.

![Sample Image Overlay Result](/Results/Image_Result.PNG)

![Sample Cube Overlay Result](/Results/Cube_Result.PNG)

The projected cube in the video stream will appear to wobble, especially when the tag is more warpped in the frame. This is probably due to the simplistic method used to compute the project matrix (the scale factor lambda is simply taken as the average of two vector magnitudes). A more complex method can be used in future iterations.

It was also noticed that the applications lose tracking for an abnormal amount of time during the latter half of Tag1.mp4 in sample_videos, even when the AR tag remains clear in the frame. This will need to be investigated further.

