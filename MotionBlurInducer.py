import cv2
import numpy as np

def add_motion_blur(image, kernel_size):
    # Create the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Apply the kernel to the input image
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred

# Load your image
image = cv2.imread('TestImage.jpg')

# Apply motion blur with a specified kernel size
blurred_image = add_motion_blur(image, 200)

# Save or display the blurred image
cv2.imwrite('blur.jpg', blurred_image)