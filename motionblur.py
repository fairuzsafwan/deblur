import cv2 
import numpy as np 
import os
from tqdm import tqdm

import time
import random

def resizeImage(img, target_size):
    h, w = img.shape[:2]
    aspect_ratio = w/h

    if w > h:
        new_w = int(target_size * aspect_ratio)
        new_h = target_size
    else:
        new_h = int(target_size / aspect_ratio)
        new_w = target_size
    
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #crop image to target_size
    half_cropSize = int(target_size/2)

    h_, w_ = resized_image.shape[:2]
    coor_h_ = int(h_ / 2)
    coor_w_ = int(w_ / 2)

    resized_image = resized_image[coor_h_-half_cropSize:coor_h_+half_cropSize, coor_w_-half_cropSize:coor_w_+half_cropSize]

    return resized_image

def processGT(imgPath, outputPath_gt, image_size):
    for path in tqdm(imgPath, "GT Progress"):
        img = cv2.imread(path)
        h, w = img.shape[:2]

        if (h < image_size) or (w < image_size):
            continue

        if (h > image_size) and (w > image_size):
            img = resizeImage(img, image_size)

        outputPath_ = os.path.join(outputPath_gt, os.path.basename(path))
        cv2.imwrite(outputPath_, img)

def motionblur(imgPath, outputPath, image_size):
    for path in tqdm(imgPath, "Train Progress"):
        img = cv2.imread(path) 

        # The greater the size, the more the motion. 
        kernel_size = random.randint(10, 20) #10
        
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

        h, w = img.shape[:2]

        if (h < image_size) or (w < image_size):
            continue

        if (h > image_size) and (w > image_size):
            #resize image
            img = resizeImage(img, image_size)
        
        # Apply the horizontal kernel. 
        horizonal_mb = cv2.filter2D(img, -1, kernel_h)

        
        # Apply the horizontal kernel. 
        horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

        outputPath_ = os.path.join(outputPath, os.path.basename(path))
        
        # Save the outputs. 
        #cv2.imwrite(outputPath_, vertical_mb)
        cv2.imwrite(outputPath_, horizonal_mb)

if __name__ == "__main__":
    datasetPath = "blur_dataset/sharp3.0"
    outputPath = "blur_dataset/motion_blurred_v3"
    outputPath_gt = "blur_dataset/sharp_v3"
    image_size = 256

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    if not os.path.exists(outputPath_gt):
        os.makedirs(outputPath_gt)

    imgPaths = [os.path.join(datasetPath, path) for path in os.listdir(datasetPath)]

    #process training dataset
    motionblur(imgPaths, outputPath, image_size)

    #process gt dataset
    processGT(imgPaths, outputPath_gt, image_size)


