import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
originalImage = cv2.imread('pattern_T03.jpeg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
imageHistogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])

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

binarization = True
if binarization:
    # Define the threshol value
    thresholdValue = 128
    # Apply binarization: pixels below thresholdValue will be 0 otherwise 255
    _, binaryImage = cv2.threshold(grayImage, thresholdValue, 255, cv2.THRESH_BINARY)
    """
    plt.figure(2)
    plt.imshow(binaryImage, cmap='gray')
    plt.title('Binary Image')
    """
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


segmentation = True
if segmentation:
   # find the contours
   contours,hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   print("Number of contours detected:",len(contours))
   for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        originalImage = cv2.drawContours(originalImage, [cnt], -1, (0,255,255), 3)
        # compute the center of mass of the triangle
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
           centroidU = int(M['m10']/M['m00'])
           centroidV = int(M['m01']/M['m00'])
           # fit an ellipse to the largest contour to find its orientation
           ellipsoidShape = cv2.fitEllipse(cnt)
           angle = ellipsoidShape[2]
           newColorImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
           cv2.circle(newColorImage, (centroidU, centroidV), 5, (0,0,255),-1)
           cv2.ellipse(newColorImage, ellipsoidShape, (0,255,0),2)
           print(f'Centroid (x,y): ({centroidU}, {centroidV})')
           print(f'Orientation angle: {angle} degrees')
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







