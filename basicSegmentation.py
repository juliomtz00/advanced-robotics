import cv2
import numpy as np
import math

# Load image
originalImage = cv2.imread('lab-project-images/undistorted-images/frame-0000.png')
# Convert image to grayscale
grayScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image (adjust threshold value as needed)
_, binaryImage = cv2.threshold(grayScaleImage, 50, 255, cv2.THRESH_BINARY)

# Find contours in the binary image to identify regions
contours, _ = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours[2]))
#print(f"Number of Contours: {len(contours)}")
# Check if there is at least one contour
if contours:
    # Take the first contour (assuming it's the only one)
    contour = contours[len(contours)-1]

    # Calculate moments for the region
    moments = cv2.moments(contour)

    # Calculate the centroid of the region
    cu = int(moments['m10'] / moments['m00'])
    cv = int(moments['m01'] / moments['m00'])
    centroid = np.array([cu,cv])
    #contourArray = np.array(contour)
    #print(contourArray[:,0][:,0])
    print(contour[:,0][:,1])
    m20 = np.mean((contour[:,0][:,0]-centroid[0])**2)
    m02 = np.mean((contour[:,0][:,1]-centroid[1])**2)
    m11 = np.mean((contour[:,0][:,0]-centroid[0])*(contour[:,0][:,1]-centroid[1]))
    # Compute the inertia matrix J
    J = np.array([[m20, m11], [m11, m02]])

    # Calculate the area of the region
    area = cv2.contourArea(contour)

    # Calculate the central moments
    #u20 = moments['mu20']-((moments['m10']**2)/ moments['m00']) 
    #u02 = moments['mu02']-((moments['m01']**2)/ moments['m00'])
    #u11 = moments['mu11']-((moments['m10']*moments['m01'])/ moments['m00'])
    # Compute the inertia matrix J
    #J = np.array([[u20, u11], [u11, u02]])

    # Calculate eigenvalues of the inertia matrix
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eigenvalues = [eigenvalues[0], eigenvalues[1]]
    
    # Determine the major eigenvector (corresponding to the larger eigenvalue)
    majorEigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    # Compute the orientation angle using atan2
    orientationAngle = math.atan2(majorEigenvector[1], majorEigenvector[0])
    print("Major EigenVector 0: ", majorEigenvector[0])
    print("Major EigenVector 1: ", majorEigenvector[1])
    # Convert the angle to degrees
    orientationAngleDeg = math.degrees(orientationAngle)+180
    # Print the properties of the single region
    print("Region:")
    print(f"Centroid: ({cu}, {cv})")
    print(f"Centroid: {centroid}")
    print(f"Area: {area}")
    print(f"Central Moments: ")
    print(f"  mu20: {m20}")
    print(f"  mu02: {m02}")
    print(f"  mu11: {m11}")
    print("Eigenvalues of Inertia Matrix J :", eigenvalues)
    print("Max EigenValue: ", np.argmax(eigenvalues))
    print("Eigenvectors of Inertia Matrix J: ")
    print("  Eigenvector 1: ", eigenvectors[:, 0])
    print("  Eigenvector 2: ", eigenvectors[:, 1])
    print("Major Eigenvector: ", majorEigenvector)
    print("Orientation Angle: ", orientationAngleDeg)
    #print(f"Moments: {moments}")

    # Draw the contour and centroid on the original image
    newColorImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(newColorImage, [contour], 0, (0, 255, 0), 2)
    cv2.circle(newColorImage, (cu, cv), 5, (0, 255, 0), -1)  # -1 fills the circle
    """
    # OPTIONAL Draw the ellipse
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(newColorImage, ellipse, (0, 0, 255), 2)
    # Extract major and minor axes from the ellipse parameters
    center, axes, angle = ellipse
    major_axis = int(max(axes) / 2)
    minor_axis = int(min(axes) / 2)
    # Calculate endpoints for major and minor axes
    major_axis_endpoint_x = int(center[0] - major_axis * math.cos(math.radians(orientationAngleDeg)))
    major_axis_endpoint_y = int(center[1] - major_axis * math.sin(math.radians(orientationAngleDeg)))

    minor_axis_endpoint_x = int(center[0] + minor_axis * math.sin(math.radians(orientationAngleDeg)))
    minor_axis_endpoint_y = int(center[1] - minor_axis * math.cos(math.radians(orientationAngleDeg)))
    # Draw major and minor axes
    cv2.line(newColorImage, (int(center[0]), int(center[1])), (major_axis_endpoint_x, major_axis_endpoint_y), (255, 0, 0), 2)
    cv2.line(newColorImage, (int(center[0]), int(center[1])), (minor_axis_endpoint_x, minor_axis_endpoint_y), (255, 0, 0), 2)
    """
    # Display the original image with contour and centroid
    resizedNewColorImage = cv2.resize(newColorImage, (800, int(800 / newColorImage.shape[1] * newColorImage.shape[0])))
    cv2.imshow('Image with Contour and Centroid', resizedNewColorImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found.")