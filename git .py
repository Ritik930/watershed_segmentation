import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('C:/Users/RITIK/OneDrive/Desktop/op.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold
threshold_value = 127
max_value = 255
_, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)

# Apply watershed segmentation
_, markers = cv2.connectedComponents(thresholded_image)
markers = markers + 1
markers[thresholded_image == 0] = 0
segmented_image = cv2.watershed(image, markers)
image_after_watershed = np.copy(image)
image_after_watershed[segmented_image == -1] = [0, 0, 255]  # Mark watershed regions in red

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate stone blast sizes
stone_blast_sizes = [cv2.contourArea(contour) for contour in contours]

# Create a figure with subplots
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

# Display the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# Display the image after watershed segmentation
axs[1].imshow(cv2.cvtColor(image_after_watershed, cv2.COLOR_BGR2RGB))
axs[1].set_title('Image after Watershed Segmentation')

# Display the thresholded image
axs[2].imshow(thresholded_image, cmap='gray')
axs[2].set_title('Thresholded Image')

# Plot the bar graph of stone blast sizes
axs[3].bar(range(1, len(stone_blast_sizes) + 1), stone_blast_sizes, color='blue')
axs[3].set_title('Stone Blast Sizes')
axs[3].set_xlabel('Stone Blast')
axs[3].set_ylabel('Size')

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
