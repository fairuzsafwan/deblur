import cv2 
import numpy as np 
import os
from tqdm import tqdm

def motionblur(imgPath, outputPath):
    for path in tqdm(imgPath, "Progress"):
        img = cv2.imread(path) 
        
        # Specify the kernel size. 
        # The greater the size, the more the motion. 
        kernel_size = 100
        
        # Create the vertical kernel. 
        #kernel_v = np.zeros((kernel_size, kernel_size)) 
        
        # Create a copy of the same for creating the horizontal kernel. 
        kernel_h = np.zeros((kernel_size, kernel_size)) 
        
        # Fill the middle row with ones. 
        #kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        
        # Normalize. 
        #kernel_v /= kernel_size 
        kernel_h /= kernel_size 
        
        # Apply the vertical kernel. 
        #vertical_mb = cv2.filter2D(img, -1, kernel_v)
        
        # Apply the horizontal kernel. 
        horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

        outputPath_ = os.path.join(outputPath, os.path.basename(path))
        
        # Save the outputs. 
        #cv2.imwrite(outputPath_, vertical_mb)
        cv2.imwrite(outputPath_, horizonal_mb)

if __name__ == "__main__":
    datasetPath = "blur_dataset/sharp"
    outputPath = "blur_dataset/motion_blurred_v2"

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    imgPaths = [os.path.join(datasetPath, path) for path in os.listdir(datasetPath)]

    motionblur(imgPaths, outputPath)
