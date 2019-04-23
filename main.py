"""
    This program demonstrates how to use a single camera for depth sensing. It uses the triangle similarity method
    to calculate the distance between a detected face and the camera. To use the program, the focal length of the
    camera must first be calibrated. Then, the facial detection algorithm can be run and the distance properly
    calculated. This program uses OpenCV for object detection.

    NOTE: Calibration can be done with an 8.5x11 piece of paper

    Author: Brandon Michelsen
    Date: 4/22/2019
"""

""" Import necessary modules """
from os import system # Import the system function for pausing the program and clearing the screen
import numpy as np # Import Numpy for math and array operations
import cv2 # Import OpenCV for computer vision operations

""" Define Global Variables """

# This variable represents the distance between the calibration object and the camera (in inches)
# Change its value if the calibration object is closer to or farther from the camera
CALIBRATE_DIST = 24

# This variable represents the width of the calibration object (in inches)
# Change its value if the calibration object's width is smaller or greater
CALIBRATE_WIDTH = 11

# This variable represents the average width of a human's face at the cranium (in inches)
# It is used in the facial distance calculations
AVG_FACE_WIDTH = 12

""" End Global Variables """

""" Define Functions """

# This function is used for calibrating the focal length of the camera
# It finds the calibration object and returns it's dimensions
# It takes an OpenCV capture object as a parameter
def find_calibrate(capt):
    # While the capture is open
    while (capt.isOpened()):
        # Read a frame from the capture device
        ret, img = capt.read()
        # If a frame exists...
        if ret is True:
            # Flip the image
            img = cv2.flip(img, 1)

            # Blur the image to filter out noise
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            blur = cv2.bilateralFilter(blur, 9, 75, 75)
            blur = cv2.medianBlur(blur, 11)

            # Use the HSV color scheme to detect only white objects and filter out other colors
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            sensitivity = 12 # Sensitivity to white colors
            lower_white = np.array([0, 0, 255-sensitivity], dtype=np.uint8) # Lower white range
            upper_white = np.array([255, sensitivity, 255], dtype=np.uint8) # Upper white range
            white = cv2.inRange(hsv, lower_white, upper_white) # Filter based on the ranges
            white = cv2.bitwise_and(img, img, mask=white)

            # Find the edges of all the white objects detected
            edged = cv2.Canny(white, 35, 125)

            # Get the contours of all the white objects
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            contours = np.array(contours)

            # Find the areas of all the white objects and store in a list of tuples
            areas = [(cv2.contourArea(c), c) for c in contours]

            """ Find the contour with the maximum area (the calibration object) """
            max_area = 0.0 # Variable for the maximum area
            max_contour = np.zeros(shape=(5, 2)) # Variable for the maximum contour
            for area in areas: # Loop through all the contour areas
                if area[0] > max_area: # Find the maximum area and grab the corresponding contour
                    max_contour = area[1]
                    max_area = area[0]

            # Display visual feedback to the user
            img = cv2.drawContours(img, [max_contour.astype(int)], -1, (0, 255, 0), 3) # Draw the maximum contour
            cv2.imshow("Contours", img) # Show the visual feedback

            # Keep running until the 'p' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('p'):
                break
        else:
            break
    cv2.destroyAllWindows() # Close all the windows
    return cv2.minAreaRect(max_contour.astype(int)) # Return the dimensions of the maximum contour

# Function to calculate the focal length of the camera
# Calls the find_calibrate function and uses the returned values to calculate the focal length
# Returns the focal length, takes in an OpenCV capture object as a parameter
def calibrate_focal_len(capt):
    global CALIBRATE_DIST # Get the global calibration distance
    global CALIBRATE_WIDTH # Get the global calibration width

    calibrate_img = find_calibrate(capt) # Find the calibration object and store its dimensions
    return (calibrate_img[1][0] * CALIBRATE_DIST) / CALIBRATE_WIDTH # Calculate the focal length

# Function to calculate the distance between a face and the camera
# Takes a known width (k_width), a given width (g_width), and focal length (focal_len)
# as parameters
def find_face_dist(k_width, g_width, focal_len):
    return (k_width * focal_len) / g_width

# Function to detect faces and find the distance between them and the camera
# Takes an OpenCV capture object, a facial recognition data set, and focal length
# as parameters
def find_dist(capt, face_dat, focal_len):
    global AVG_FACE_WIDTH # Get the global average face width

    # While the capture device is open
    while (capt.isOpened()):
        # Read a frame from the capture device
        ret, frame = capt.read()

        # If a frame is read
        if ret is True:
            # Flip the image
            frame = cv2.flip(frame, 1)

            # Apply a gray filter for filtering noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect any faces in the frame
            faces_front = face_dat.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # For all the face coordinates in the frame
            for (x, y, w, h) in faces_front:
                frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2) # Draw a rectangle around the face
                dist = find_face_dist(AVG_FACE_WIDTH, w, focal_len) # Detect the distance between the camera and the face
                text = "Dist: {:.2f}".format(dist) # Dispaly the distance to the face in the frame
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)

            # Show the resulting frame
            cv2.imshow('Frame', frame)

            # Run until the user presses the 'q' key
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # Close out all the windows
    cv2.destroyAllWindows()

# Function for printing a description of the program to the user
def print_directions():
    # Print a description
    print("This program provides a method of depth sensing with a univision camera (rather than stereovision).")
    print("The program uses a method known as triangle similarity to find the distance from an object to the camera.")
    print("In this case, the object that the camera is tracking is a human face. The program will calculate the distance"
          " to a detected face based on the average width of a human face.")
    print("Depth sensing is very useful to many fields that use computer vision, due to the fact that sensing depth"
          " allows a machine to detect where at in an image an object is.")
    print("This program serves as the first part of a depth sensing camera with only one lens.")

    print("\nTo use the application, the camera must first be calibrated. To calibrate, select option 1 from the menu.")
    print("A screen will pop up giving you visual feedback about the contours of bright objects.")
    print("Next, hold a white object that is 11 inches in width at a distance of 24 inches from the camera.")
    print("Once the white object is in place, press the 'p' key to snap a picture of the object.")
    print("The program will then calculate the focal length of the camera.")

    print("\nOnce the focal length has been calculated, the main part of the program can be run.")
    print("To run the face tracking algorithm, select option 2 from the menu.")
    print("A window will appear showing what the camera sees, indicating any faces as well as the distance"
          " to a detected face.")
    print("To close out of the main part of the program, press the 'q' key.")

# The main function of the program
def main():
    """ Define Local Variables """
    focal = 0.0 # Variable for the calibrated focal length of the camera
    opt = '' # Function for storing the user's option from the menu

    # Train facial detection algorithm using a cascade classifier
    face_frontal = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml')

    # Get the first connected capture device
    capture = cv2.VideoCapture(0)

    # If no capture device is detected, close the program
    if not capture.isOpened():
        print("ERROR: Camera could not be detected")
    # Otherwise, run the program
    else:
        # While the capture device is available and the user has not quit
        while capture.isOpened() and opt != '0':
            # Print the menu
            print("\tWelcome to the Camera Distance Calculator!")
            print("\nPlease select an option from the menu:")
            print("1) Calibrate the camera")
            print("2) Run real-time distance calculations")
            print("3) View the directions")
            print("0) Quit the program\n")

            # Get the user's choice
            opt = input("Please select an option from the menu: ")

            system('cls') # Clear the screen

            # If the user entered option 1, calibrate the camera
            if opt == '1':
                print("Press 'p' to finish the calculation...")
                focal = calibrate_focal_len(capture)
                print("Focal Length: {:.2f}".format(focal)) # Display the focal length to 2 decimal places
            # If the user entered option 2, run the facial detection program
            elif opt == '2':
                print("Press 'q' to quit the calculation...")
                find_dist(capture, face_frontal, focal)
            # If the user entered option 3, display the description
            elif opt == '3':
                print_directions()
            # If the user entered option 0, continue on to quit
            elif opt == '0':
                continue
            # If the user enters anything else, tell them it is not a valid option
            else:
                print("That is not a valid option. Please try again.")

            system('pause') # Pause the program
            system('cls') # Clear the screen

    # Release the capture device
    capture.release()

    # Close all windows
    cv2.destroyAllWindows()

# Run the main program if this is the main class
if __name__ == "__main__":
    main()