import cv2
import numpy as np

# Signos de los cuadrantes naturalmente positivos, solo se transforma para eje X
# signQuadrantX = 1
# ------------------------------------------------------------------------------
# LO ANTERIOR SOLO APLICA PARA UN EJE COORDENADO NORMAL, NO EL QUE DA EL CV2
# Para el cv2 el cuadrante que se transforma sera la del eje Y
angle_values = []

# Reading an image using OpenCV, extracting its height and width, and converting it to grayscale.
# Create a new window
cv2.namedWindow("Camera Frame", cv2.WINDOW_NORMAL)

# Create a video capture object for video streaming
camera_index = 0 # this is the camera index
video_capture = cv2.VideoCapture(camera_index)
video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off

# Calculate the image measurements of the camera through the functions of the opencv library.
image_width = float(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = float(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("IMAGE MEASUREMENTS: height: "+str(image_height)+", width: "+str(image_width))

# Open the calibration file
calibration_file = open("rob-lab-calib/calibration-parameters.txt", 'r')

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

# If camera opens correctly
i = 0

while(video_capture.isOpened()):

    #Get the current frame and pass it on to 'frame''
    #If the current frame cannot be captured, ret = 0
    ret,frame = video_capture.read()

    

    # If ret=0
    if not ret:
        print("Frame missed!")

    # Get size
    h,  w = frame.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort image
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Convert image to gray
    grayImage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

    # Applying binarization to the grayscale image using a threshold value.
    binarization = True
    if binarization:
        # Define the threshol value
        thresholdValue = 128
        # Convert the original image to the HSV color space for better color segmentation
        hsvImage = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the green color in HSV
        lower_green = np.array([50, 100, 100])  # Adjust these values as needed
        upper_green = np.array([70, 255, 255])

        # Create a mask to filter the green color
        green_mask = cv2.inRange(hsvImage, lower_green, upper_green)

        # Combine the grayscale image with the green mask
        combined_mask = cv2.bitwise_or(grayImage, green_mask)

        #Gamma
        c = 1
        gamma_value = 25.0
        rn = combined_mask/255
        img_gamma = c*(rn**gamma_value)
        img_proc = np.uint8((255/(np.max(img_gamma)-np.min(img_gamma)))*(img_gamma-np.min(img_gamma)))

        # Applying binarization to the combined mask using a threshold value.
        _,binaryImage = cv2.threshold(img_proc, thresholdValue, 255, cv2.THRESH_BINARY_INV)
        
    segmentation = True
    if segmentation:
        # find the contours
        # contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # SE CAMBIO EL ULTIMO ARGUMENTO A "cv2.CHAIN_APPROX_NONE" para corregir el error con "pattern_T01.jpeg"
        contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        # Iterate through the contours and filter out small ones
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        print("Number of filtered contours detected:", len(filtered_contours))
        
        fIteration = len(filtered_contours)
        filtered_last = filtered_contours[fIteration-1]
        approx = cv2.approxPolyDP(filtered_last, 0.02*cv2.arcLength(filtered_last, True), True)
        
        # Check if there are enough points to fit an ellipse
        if len(filtered_last) == 4: 
            average_point = np.mean(filtered_last[:, 0], axis=0, dtype=np.int32)
            filtered_last = np.vstack([filtered_last[:, 0], average_point.reshape(1, 2)])

        if len(filtered_last) >= 5:
            # compute the center of mass of the triangle
            originalImage = cv2.drawContours(dst, [filtered_last], -1, (0,255,255), 3)
            # compute the center of mass of the triangle
            M = cv2.moments(filtered_last)
            '''for num in cnt:
                print(num)
            #print(M)'''
            if M['m00'] != 0.0:
                centroidU = int(M['m10']/M['m00'])
                centroidV = int(M['m01']/M['m00'])
                # fit an ellipse to the largest contour to find its orientation
                ellipsoidShape = cv2.fitEllipse(filtered_last)
                # ellipsoidShape[0]  ->  (centroideCoordX, centroideCoordY)
                # ellipsoidShape[1]  ->  (centroideWidth, centroideHeight)
                # ellipsoidShape[2]  ->  Angulo que genera la elipse, los ejes no corresponden a los de un eje cordenado
                #                        normal, su X corresponde al eje Y (normal), y, su Y corresponde al eje X (normal)
                angle = ellipsoidShape[2]
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                angle_values.append(angle)
                if len(angle_values) > 10:
                    angle_values.pop(0)
                
                average_angle = np.sum(angle_values)/len(angle_values)
                cv2.circle(dst, (centroidU, centroidV), 5, (0, 0, 255), -1)   # RED
                #cv2.ellipse(dst, ellipsoidShape, (0, 255, 0), 2)              # GREEN
                print(f'Centroid (x,y): ({centroidU}, {centroidV})')
                print(f'Orientation angle: {angle} degrees')
                # -------------------------------------------------------------
                # AGREGADO
                # -------------------------------------------------------------
                cv2.drawContours(dst, [filtered_last], -1, (0, 255, 255), 3)         # YELLOW
                # -------------------------------------------------------------
                 # If so, it is then visualised
                cv2.putText(dst, f'Angle:{round(average_angle,3)}', (centroidU, centroidV), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Camera Frame", dst)
            else:
                print('Error')