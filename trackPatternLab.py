import cv2
import numpy as np
import matplotlib.pyplot as plt

# Signos de los cuadrantes naturalmente positivos, solo se transforma para eje X
# signQuadrantX = 1
# ------------------------------------------------------------------------------
# LO ANTERIOR SOLO APLICA PARA UN EJE COORDENADO NORMAL, NO EL QUE DA EL CV2
# Para el cv2 el cuadrante que se transforma sera la del eje Y
signQuadrantX = 1
signQuadrantY = signQuadrantX

# Reading an image using OpenCV, extracting its height and width, and converting it to grayscale.
originalImage = cv2.imread('lab-project-images/undistorted-images/frame-0002.png')
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
if binarization:
    # Define the threshol value
    thresholdValue = 128
    # Apply binarization: pixels below thresholdValue will be 0 otherwise 255
    _, binaryImage = cv2.threshold(grayImage, thresholdValue, 255, cv2.THRESH_BINARY_INV)
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
        fileName = "/home/gil/Python101/sensors/computerVision/binaryImage.png"
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

#plt.show()


#coordX


segmentation = True
if segmentation:
   # find the contours
   #contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # SE CAMBIO EL ULTIMO ARGUMENTO A "cv2.CHAIN_APPROX_NONE" para corregir el error con "pattern_T01.jpeg"
   contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   print("Number of contours detected:",len(contours))
   for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
    # Check if there are enough points to fit an ellipse
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
                cv2.circle(newColorImage, (centroidU, centroidV), 5, (0,0,255),-1)   # RED
                cv2.ellipse(newColorImage, ellipsoidShape, (0,255,0),2)              # GREEN
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