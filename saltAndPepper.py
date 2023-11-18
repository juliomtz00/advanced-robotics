import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Gaussian filter
def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Function to apply Median filter
def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# Function to display image and its histogram
def show_image_and_histogram(image, title):
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title(title)
    plt.subplot(122), plt.hist(image.ravel(), 256, [0, 256]), plt.title('Histogram')
    plt.show()

# Load the original image
image_path = 'CRGSNoise.jpeg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian filter
gaussian_filtered_image = apply_gaussian_filter(original_image)

# Apply Median filter
median_filtered_image = apply_median_filter(original_image)

# Display the results
show_image_and_histogram(original_image, 'Original Image')
show_image_and_histogram(gaussian_filtered_image, 'Gaussian Filtered Image')
show_image_and_histogram(median_filtered_image, 'Median Filtered Image')