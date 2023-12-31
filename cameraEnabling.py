'''
get-measurements.py

USAGE:
python3 get-measurements.py --cam_index 1 --Z 1.04 

    Obtains a real world measurement between two selected points chosen from the camera view, with a minimal error between them.
    The focal length must be known or aproximated in order to obtain a more accurate value.

Authors:
+ Julio Enrique Martinez Robledo- julio.martinezr@udem.edu

Institution: Universidad de Monterrey
Subject: Computational Vision
Lecturer: Dr. Andrés Hernández Gutiérrez

Date of creation: March 3rd 2023
Last update: March 7th 2023
'''

# Import needed libraries in the code.
import numpy as np
import cv2
import argparse

# Parse user's arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--cam_index', type=int, default=1,help="Index value for the Camera")
parser.add_argument("--path_to_saved_images",
                    type=str,
                    default='captured-images',
                    help='Folder where the captured images will be stored')
#parser.add_argument('--cal_file', type=str, help="Name of the calibration file")
args = parser.parse_args()
    
# Create a new window
cv2.namedWindow("Camera Frame", cv2.WINDOW_NORMAL)

# Create a video capture object for video streaming
camera_index = args.cam_index
video_capture = cv2.VideoCapture(camera_index)
video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off

# Calculate the image measurements of the camera through the functions of the opencv library.
image_width = float(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = float(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("IMAGE MEASUREMENTS: height: "+str(image_height)+", width: "+str(image_width))

# If camera opens correctly
i = 0
while(video_capture.isOpened()):
    #Get the current frame and pass it on to 'frame''
    #If the current frame cannot be captured, ret = 0
    ret,frame = video_capture.read()

    # If ret=0
    if not ret:
        print("Frame missed!")

     # If so, it is then visualised
    cv2.imshow("Camera Frame", frame)
    
    # Retrieve the pressed key
    key = cv2.waitKey(1)

    # If the pressed key was 's''
    #the current image is saved into disk
    if key == ord('s'):
        path_and_image_name = str(args.path_to_saved_images)+'frame'+"-"+str(f"{i:04d}.png")
        print('Captured image saved in: ' + path_and_image_name)
        cv2.imwrite(path_and_image_name, frame)
        i += 1
    
    # If the pressed key was  'd''
    # the program finishes
    elif key == ord("q"):
        break
        
#Close video capture object
video_capture.release()

#Close all windows
cv2.destroyAllWindows()