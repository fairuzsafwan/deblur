import os
import cv2
import numpy as np

def apply_motion_blur(image_path, kernel_size=15):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was read successfully
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return None
    
    # Define the motion blur kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    
    # Apply the motion blur kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    
    return blurred_image

def main(folder_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get a list of all JPG files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # Apply motion blur to each image and save it in the output folder
    for filename in jpg_files:
        input_image_path = os.path.join(folder_path, filename)
        output_image_path = os.path.join(output_folder, filename)
        
        blurred_image = apply_motion_blur(input_image_path)
        
        if blurred_image is not None:
            cv2.imwrite(output_image_path, blurred_image)
            # print(f"Motion blur applied to '{filename}' and saved as '{output_image_path}'.")

if __name__ == "__main__":
    # Path to the folder containing JPG images
    input_folder_path = "blur_dataset/sharp"  
    
    # Output folder where blurred images will be saved
    output_folder_path = "blur_dataset/motion_blurred" 
    
    main(input_folder_path, output_folder_path)
