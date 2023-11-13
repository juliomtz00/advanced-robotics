# Import standard libraries
import numpy as np
import cv2 as cv
import glob
import argparse
import os
import textwrap

# Parse user's argument
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
        This Python script performs the calibration process for a monocular
        camera. It uses a chessboard calibration panel to estimate both the
        camera matrix and the lens distortions needed to subsequently undis-
        tor any other image acquired by that particular camera.

        '''))
parser.add_argument("--path_to_distorted_images",
                    type=str,
                    default='distorted-images',
                    help='Folder where the testing distorted images are')
parser.add_argument("--path_to_undistorted_images",
                    type=str,
                    default='undistorted-images',
                    help='Folder where the undistorted images will be saved')
parser.add_argument("--path_to_calibration_file",
                    type=str,
                    default='calibration_file',
                    help='Folder where the calibration file is')
args = parser.parse_args()

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# Path to undistorted images
path_to_undistorted_images = args.path_to_undistorted_images

# Open the calibration file
calibration_file = open(args.path_to_calibration_file, 'r')

# Initialize variables to store mtx and dist
mtx = None
dist = None

# Loop through each line in the calibration file
for line in calibration_file:
    line = line.rstrip()

    # Check if the line contains information about mtx
    if line.startswith('mtx:'):
        # Extract the matrix values from the line and convert them to a NumPy array
        mtx_values = np.array(eval(line[4:]))
        mtx = mtx_values.reshape((3, 3))  # Assuming mtx is a 3x3 matrix

    # Check if the line contains information about dist
    elif line.startswith('dist:'):
        # Extract the matrix values from the line and convert them to a NumPy array
        dist_values = np.array(eval(line[5:]))
        dist = dist_values.reshape((1, -1))  # Assuming dist is a 1xN matrix

# Close the calibration file
calibration_file.close()


print("mtx:{}".format(mtx))
print("dist:{}".format(dist))
print("Camera calibration completed!")



print("""
# ------------------------------------------------------------------- #
# ---------------- UNDISTORT IMAGES --------------------------------- #
# ------------------------------------------------------------------- #
""")


# Load calibration images
path_to_distorted_images=args.path_to_distorted_images
images = glob.glob(path_to_distorted_images+'*.png')

# Loop through distorted images
for fname in images:

    print("Undistorting: {}".format(fname))
    img_names = fname.split('/')[-1]

    # read current distorted image
    img = cv.imread(fname)

    # Get size
    h,  w = img.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(path_to_undistorted_images+img_names, dst)
    print("Undistorted image saved in:{}".format(path_to_undistorted_images+img_names))