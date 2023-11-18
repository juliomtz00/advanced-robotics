import cv2
import numpy as np

def calculate_moments(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assumes the number is the largest object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate moments
    moments = cv2.moments(largest_contour)

    # Calculate central moments
    central_moments = cv2.moments(largest_contour, True)

    # Calculate Hu moments
    hu_moments = cv2.HuMoments(central_moments)

    return moments, central_moments, hu_moments.flatten()

def classify_number(test_hu_moments, reference_moments):
    # Compare Hu moments using average Euclidean distance
    distances = {label: np.mean(np.abs(test_hu_moments - ref_hu_moments)) for label, ref_hu_moments in reference_moments.items()}

    # Find the label with the smallest distance
    min_label = min(distances, key=distances.get)

    return min_label  # Return the label as the classified number (1, 2, or 3)

# Load reference images
number_1 = cv2.imread('pattern_N1.png')
number_2 = cv2.imread('pattern_N2.png')
number_3 = cv2.imread('pattern_N3.png')

# Calculate moments for reference images
moments_1, central_moments_1, hu_moments_1 = calculate_moments(number_1)
moments_2, central_moments_2, hu_moments_2 = calculate_moments(number_2)
moments_3, central_moments_3, hu_moments_3 = calculate_moments(number_3)

# Create a dictionary to store reference moments
reference_moments = {
    1: hu_moments_1,
    2: hu_moments_2,
    3: hu_moments_3
}

# Load the new image for testing
new_image = cv2.imread('pattern_VN.png')

# Calculate moments for the new image
test_moments, test_central_moments, test_hu_moments = calculate_moments(new_image)

# Classify regions in the new image
classified_numbers = []
for i in range(test_hu_moments.shape[0]):
    classified_numbers.append(classify_number(test_hu_moments[i], reference_moments))

print("Classified Numbers:", classified_numbers)

