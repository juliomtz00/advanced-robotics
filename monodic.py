import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--save_image", 
                    type=int, 
                    default=0,
                    help="Folder where the calibration images are")
args = parser.parse_args()

originalImage = cv2.imread('CRGS.jpeg')
greyImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
imageHistogram = cv2.calcHist([greyImage],[0],None,[256],[0,256])

plt.figure(1)
plt.subplot(1,3,1)
colorImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2RGB)
plt.imshow(colorImage)
plt.title("Color Image")
plt.subplot(1,3,2)
plt.imshow(originalImage)
plt.imshow(greyImage,cmap="gray")
plt.title("Greyscale Image")
plt.subplot(1,3,3)
plt.hist(greyImage.ravel(),256,[0,256])
plt.xlim([0,256])
plt.tight_layout()
plt.title("Histogram")
#plt.show()

binarization = True
if binarization:
    #Define threshold value
    upperThreshold = 180
    lowerThreshold = 150
    #Apply binarization: pixels below threshold value will be 0 otherwise 255
    _, binaryImage = cv2.threshold(greyImage,lowerThreshold,upperThreshold,cv2.THRESH_BINARY)

    plt.figure(2)
    plt.imshow(binaryImage,cmap="gray")
    plt.title("Binary Image")
    saveImage = args.save_image
    if saveImage == 1:
        fileName = "binaryImage.png"
        cv2.imwrite(fileName,binaryImage)
        print("Binary Image saved")

plt.axis('off')
plt.show()