# AR Video Overlay Application

## Overview




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

To run code for Part 1, use:
```
python3 ARTagDecoder.py
```
The projection matrix, tag corners (corrected for orientation) and tag ID in sample.png will be shown


To run the code for Part 2a, use:
```
python3 ImageOverlay.py
```
The video of the Lena image overlaid on the desired AR tag in the correct orientation will be shown. The desired tag ID can be changed at the bottom of the script.


To run the code for Part 2b, use:
```
python3 CubeOverlay.py
```
As with the previous part, the desired tag ID can be changed at the bottom of the script.


## Notes and Known Issues
