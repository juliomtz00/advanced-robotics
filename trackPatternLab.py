import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Signos de los cuadrantes naturalmente positivos, solo se transforma para eje X
# signQuadrantX = 1
# ------------------------------------------------------------------------------
# LO ANTERIOR SOLO APLICA PARA UN EJE COORDENADO NORMAL, NO EL QUE DA EL CV2
# Para el cv2 el cuadrante que se transforma sera la del eje Y
signQuadrantX = 1
signQuadrantY = signQuadrantX

# Reading an image using OpenCV, extracting its height and width, and converting it to grayscale.
# Path to undistorted images
path_to_undistorted_images = 'lab-project-images/undistorted-images/*.png'
t


# Load calibration images
images = glob.glob(path_to_undistorted_images)

for fname in images:

    originalImage = cv2.imread(fname)

    height, width = originalImage.shape[:2]
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    # Displaying the grayscale image and calculating its histogram.
    cv2.imshow("Shapes", grayImage)
    imageHistogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    """
    plt.figure(1)
    plt.subplot(1,3,1)
    colorImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    plt.imshow(colorImage)
    plt.title('Color Image')
    plt.subplot(1,3,2)
    plt.imshow(originalImage)
    plt.imshow(grayImage, cmap='gray')
    plt.title('Grayscale Image')
    plt.subplot(1,3,3)
    plt.hist(grayImage.ravel(), 256,[0,256])
    plt.xlim([0,256])
    plt.tight_layout()
    plt.title('Histogram')
    """

    # Applying binarization to the grayscale image using a threshold value.
    binarization = True
    img_names = fname.split('/')[-1]
    print(f"""\n# ------------------------------------------------------------------- #
    # --------------- LOADING IMAGE {img_names} -------------------------------- #
    # ------------------------------------------------------------------- #""")
    if binarization:
        # Define the threshol value
        thresholdValue = 128
        # Convert the original image to the HSV color space for better color segmentation
        hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the green color in HSV
        lower_green = np.array([40, 100, 100])  # Adjust these values as needed
        upper_green = np.array([80, 255, 255])

        # Create a mask to filter the green color
        green_mask = cv2.inRange(hsvImage, lower_green, upper_green)

        # Combine the grayscale image with the green mask
        combined_mask = cv2.bitwise_or(grayImage, green_mask)

        # Display the combined mask
        cv2.imshow("Combined Mask", combined_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Gamma
        c = 1
        gamma_value = 25.0
        rn = combined_mask/255
        img_gamma = c*(rn**gamma_value)
        img_proc = np.uint8((255/(np.max(img_gamma)-np.min(img_gamma)))*(img_gamma-np.min(img_gamma)))

        # Applying binarization to the combined mask using a threshold value.
        _,binaryImage = cv2.threshold(img_proc, thresholdValue, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("Binary Image", binaryImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        plt.figure(2)
        #plt.imshow(binaryImage, cmap='gray')
        newColorImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Binary Image', newColorImage)
        #plt.title('Binary Image')
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        '''
        saveImage = False
        if saveImage:
            fileName = "binaryImage.png"
            cv2.imwrite(fileName,binaryImage)
            print("Binary Image Saved")

        """
        invertedBinaryImage = True
        if invertedBinaryImage:
            invertBinaryImage = cv2.bitwise_not(binaryImage)
            
            plt.figure(3)
            plt.imshow(invertBinaryImage, cmap='gray')
            plt.title('Inverted Binary Image')
        """
        
        
    segmentation = True
    if segmentation:
        # find the contours
        contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # SE CAMBIO EL ULTIMO ARGUMENTO A "cv2.CHAIN_APPROX_NONE" para corregir el error con "pattern_T01.jpeg"
        #contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print("Number of contours detected:",len(contours))
        
        # Iterate through the contours and filter out small ones
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        print("Number of filtered contours detected:", len(filtered_contours))
        
        cnt = filtered_contours[1]
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        # if len(approx) == 3:
        
        # Check if there are enough points to fit an ellipse
        if len(cnt) == 4: 
            average_point = np.mean(cnt[:, 0], axis=0, dtype=np.int32)
            cnt = np.vstack([cnt[:, 0], average_point.reshape(1, 2)])

        if len(cnt) >= 5:
            # compute the center of mass of the triangle
            originalImage = cv2.drawContours(originalImage, [cnt], -1, (0,255,255), 3)
            # compute the center of mass of the triangle
            M = cv2.moments(cnt)
            '''for num in cnt:
                print(num)
            #print(M)'''
            if M['m00'] != 0.0:
                centroidU = int(M['m10']/M['m00'])
                centroidV = int(M['m01']/M['m00'])
                # fit an ellipse to the largest contour to find its orientation
                ellipsoidShape = cv2.fitEllipse(cnt)
                # ellipsoidShape[0]  ->  (centroideCoordX, centroideCoordY)
                # ellipsoidShape[1]  ->  (centroideWidth, centroideHeight)
                # ellipsoidShape[2]  ->  Angulo que genera la elipse, los ejes no corresponden a los de un eje cordenado
                #                        normal, su X corresponde al eje Y (normal), y, su Y corresponde al eje X (normal)
                angle = ellipsoidShape[2]
                # -------------------------------------------------------------
                # AJUSTE DEL ANGULO [INCOMPLETO]
                # -------------------------------------------------------------
                ang = signQuadrantY
                # -------------------------------------------------------------
                newColorImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
                cv2.circle(newColorImage, (centroidU, centroidV), 5, (0, 0, 255), -1)   # RED
                cv2.ellipse(newColorImage, ellipsoidShape, (0, 255, 0), 2)              # GREEN
                print(f'Centroid (x,y): ({centroidU}, {centroidV})')
                print(f'Orientation angle: {angle} degrees')
                # -------------------------------------------------------------
                # AGREGADO
                # -------------------------------------------------------------
                cv2.drawContours(newColorImage, [cnt], -1, (0, 255, 255), 3)         # YELLOW
                # -------------------------------------------------------------
                cv2.imshow("Shapes", newColorImage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #cv2.putText(originalImage, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imwrite(path_to_created_images+img_names,newColorImage)
                print("Undistorted image saved in:{}".format(path_to_created_images+img_names))

            else:
                print('Error')

"""
# === COMPUTE MIDPOINT FROM SHORTEST TRIANGLE SIDE ===
# Find the vertices of the triangle
vertices = contours[:, 0]
# Calculate the lengths of the three sides of the triangle
sideLengths = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 3]) for i in range(3)]
# Find the index of the shortest side
shortestSideIndex = sideLengths.index(min(sideLengths))
# Calculate the midpoint of the shortest side
midPoint = (vertices[shortestSideIndex] + vertices[(shortestSideIndex + 1) % 3]) // 2
# Calculate the slope between the midpoint and the centroid
slope = (centroidV - midPoint[1]) / (centroidU - midPoint[0])
# Draw the centroid, orientation, and shortest side on the image (optional)
print(f'Midpoint (x, y): {midPoint}')
print(f'Slope: {slope:.2f}')
"""