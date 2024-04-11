import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('Image_Q3.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("a", image)

# parameters for different transformations
parameters = [0.5, 1, 10, 75]

# Apply different log transformations and display the results
for i, c in enumerate(parameters):
    t_image = c * np.log1p(image)

    # Clip values so they are within range 0 to 255
    t_image = np.clip(t_image, 0, 255).astype(np.uint8)

    # Display the images with the parameter values 
    plt.subplot(2, 2, i + 1)
    plt.imshow(t_image, cmap='gray')
    plt.title(f'c = {c}')
    plt.axis('off')

plt.suptitle('Different Log Transformations', fontsize=16)
plt.show()
