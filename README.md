# Camera-Pose-Estimation
# ENPM673 – Perception for Autonomous Robots

# Project 2
Problem 1 [70 pts]:
In this problem, you will perform camera pose estimation using homography. Given this video your
task is to compute the rotation and translation between the camera and a coordinate frame whose
origin is located on any one corner of the sheet of paper.
In order to do so, you must:
+ Design an image processing pipeline to extract the paper on the ground and then extract all
of its corners using the Hough Transformation technique .
+ Once you have all the corner points, you will have to compute homography between real
world points and pixel coordinates of the corners. You must write your own function to
compute homography.
+ Decompose the obtained homography matrix to get the rotation and translation
+ Note: If you decide to resize the image frames, you need to accordingly modify your intrinsic matrix
too. Refer to this discussion.
+ Data:
The dimensions of the paper is 21.6 cm x 27.9 cm.
The intrinsic matrix of the camera can be found here.

![Lines_Detected](https://user-images.githubusercontent.com/112987383/223968597-0bca53fc-fa3a-47a6-824a-ed909efcc81c.jpg)


## A. File Structure

This projects consists of the following code files
+ Problem #1:
    1. Problem1.py
+ Problem #2:
    1. Problem2.py

## B. Modification to the given dataset/video/image:
- None

## C. Dependancies

+ Ensure the following depenancies are installed
    ```
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install opencv-python
    ```

+ Ensure that the above programs are downloaded into the same folder containing 
'project2.avi'

## D. Running the Program

+ Problem1:
+ Run the program "Problem1.py" to check the outputs.
+ As the program gets executed, the video, with detected lines and corners, can be seen.
+ The program will run for 1-2 mins, after which the camera pose estimation plots, number of lines plots are displayed.
+ The file - "HomographY_check", also gets generated. This is a text file to cross-check the generated matrix.
+ ---------
+ Problem2:
+ Run the program "Problem2.py" to check the outputs.
+ The panoramic image stiching given four images will be displayed.

## E. Results
+ On running each of the proframs, the output either pops out a plot or a video in individual window. The outputs can be correlated with the outputs shown in the report.
