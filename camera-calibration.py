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
parser.add_argument("--path_to_calibration_images", 
                    type=str, 
                    default='calibration-images',
                    help="Folder where the calibration images are")
parser.add_argument("--path_to_colored_images", 
                    type=str, 
                    default='colored-images',
                    help="Folder where the colored images are")
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
                    help='Folder where the calibration file will be saved')
args = parser.parse_args()

def save_calib_params():
    # use os.path.join() to create the full file path
    file_path =  os.path.join(args.path_to_calibration_file, "calibration-parameters.txt")

    # open the file in write mode
    with open(file_path, "w") as f:
        # write the data to the file
        f.write("mtx:{}".format(mtx)+"\n")
        f.write("dist:{}".format(dist))


# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Path to calibration images
path_to_calibration_images = args.path_to_calibration_images

# Path to undistorted images
path_to_undistorted_images = args.path_to_undistorted_images

# Load calibration images
images = glob.glob(path_to_calibration_images+'*.png')


# Create a new window for visualisation purposes
cv.namedWindow('Current calibration image', cv.WINDOW_NORMAL)
print("""\n# ------------------------------------------------------------------- #
# ---------------  CORNERS DETECTION -------------------------------- #
# ------------------------------------------------------------------- #""")
for fname in images:

    # Indicate path and name of current calibration image
    print(f"Finding corners on {fname}...")

    # Read current calibration image
    img = cv.imread(fname)

    if img.size == 0:
        print(f"ERROR! - image {fname} does not exist")
        exit(1)

    # Convert from BGR to greyscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:

        # Find corners on chessboard
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)


        # Draw and display the corners
        cv.drawChessboardCorners(img, (9, 6), corners2, ret)

        # Resize current calibration image
        scale_percent = 100
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

        # Visualise current calibration image with detected corners
        cv.imshow('Current calibration image', img)
        path_to_colored_images=args.path_to_colored_images
        img_names = fname.split('/')[-1]
        cv.imwrite(path_to_colored_images+img_names, img)
        cv.waitKey(200)

        print("Corner detection completed!")

# Destroy all windows
cv.destroyAllWindows()


# ------------------------------------------------------------------- #
# ---------------  CAMERA CALIBRATION ------------------------------- #
# ------------------------------------------------------------------- #
print("""\n# ------------------------------------------------------------------ #
# ---------------  CAMERA CALIBRATION ------------------------------ #
# ------------------------------------------------------------------ #""")
print("Performing camera calibration...")

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print calibration parameters
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

save_calib_params()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "\nTotal error: {}".format(mean_error/len(objpoints)) )