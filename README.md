# FacialDist_OpenCV
A program to determine the distance between a camera and a detected face using OpenCV
# Summary:
This program allows a camera to detect distances between a detected face and the camera itself. It serves as a very rudimentary form of depth sensing for uni-vision camera (as opposed to stereo-vision). The program allows a user to detect faces using a simple webcam and then detect how far away from the camera that face is. This could be useful for many computer vision applications facial detection and distance sensing. One example could be a robot that detects humans and how far away they are, and maneuvers around them once the human gets too close. This is only one example, however, and there are many more to explore.
To use the program, the camera must first be calibrated. To do this, select the first option from the user menu, measure a distance of 24 inches from the camera, and hold a white object that is 11 inches in width at that distance from the camera (an 8.5 in. x 11 in. piece of paper would work). Then, press the ‘p’ key on your keyboard. This will properly calibrate the camera. After that, you can run the second option from the menu, which will find the distance from the camera to a human face in real time. Press the ‘q’ key on your keyboard when you want to stop facial detection and select the final option from the menu when you want to exit the program. 
# Dependencies:
- OpenCV version 4.0.0
- Numpy version 1.16.2
- Python OS module
- Python 3.6.1
- A webcam
# Install
-	Ensure that all the required dependencies are downloaded and installed on your computer
-	Download the ‘main.py’ file to your computer
# Run
-	Open up a command prompt
-	Navigate to the directory where the file is stored
-	Type “python main.py” or “python.exe main.py”
-	The program will then run

