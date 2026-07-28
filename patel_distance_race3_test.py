
# This file contains code that was used to test the original distance race. The original distance race, DistanceRace3 was ultimately not used
# due to difficulty in extracting the connected components. A new distance race was used, the DistanceRaceNew image set.

import os
import skimage   
from skimage import io   
import numpy as np   
import matplotlib.pyplot as plt   
from skimage import filters  
import scipy  
from skimage import feature   
from scipy import ndimage  
import pandas as pd 
from skimage.measure import label, regionprops, regionprops_table # Using imports from Homework 3


distImgFolder = "DistanceRace3" # Store the folder

distImgs = [] # Initialize storage for the images


for i in range(2, 25): # Loop through all action images in the speed race
    file = f"DIST3_{i:02d}.jpeg" # Get the current image
    path = os.path.join(distImgFolder, file) # Make the file path string
    image = io.imread(path).astype('float64') / 255.0 # Read the image
    image = np.rot90(image, -1) # Rotate image to ensure correct orientation
    image = image[900:, :, :] # Crop the top part of image out to eliminate the wall
    distImgs.append(image) # Add the image to the array of images


file = "DIST3_01.jpeg" # Get the background image
path = os.path.join(distImgFolder, file) # Make the file path string
image = io.imread(path).astype('float64') / 255.0 # Read the image
image = np.rot90(image, -1) # Rotate image to ensure correct orientation
image = image[900:, :, :] # Crop the top part of image out to eliminate the wall
background = image # Store our background image

"""
# Display some images for reference
plt.figure(figsize=(12, 8))

# First race frame
plt.subplot(2, 2, 1)
plt.imshow(distImgs[0])
plt.title("DIST3_02")
plt.axis("off")

# Middle race frame
plt.subplot(2, 2, 2)
plt.imshow(distImgs[9])
plt.title("DIST3_11")
plt.axis("off")

# Last race frame
plt.subplot(2, 2, 3)
plt.imshow(distImgs[22])
plt.title("DIST3_24")
plt.axis("off")

# Background without finish line
plt.subplot(2, 2, 4)
plt.imshow(background)
plt.title("Background")
plt.axis("off")


plt.show() # Display the reference images


"""


# Experiment with background subtraction III in region extraction slides
race = distImgs[4] # Store first race image 
difference = np.sqrt( # Perform backsub III
    (race[:,:,0] - background[:,:,0])**2 +
    (race[:,:,1] - background[:,:,1])**2 +
    (race[:,:,2] - background[:,:,2])**2
)

print("Backsub Image Dimensions: ", difference.shape) # Print shape to confirm grayscale

thresholded = difference > 0.5 # Threshold the backsub image


# Display the first race image, the background with finish line, and resulting backsub image
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(distImgs[0])
plt.title("DIST3_02")
plt.axis("off")

# Background without finish line
plt.subplot(2, 2, 2)
plt.imshow(background)      
plt.title("Background")
plt.axis("off")

# Backsub image result
plt.subplot(2, 2, 3)
plt.imshow(difference, cmap='gray')      
plt.title("Background Subtraction Result")
plt.axis("off")

# Threshold result
plt.subplot(2, 2, 4)
plt.imshow(thresholded, cmap='gray')
plt.title("Threshold = 0.5")
plt.axis("off")

plt.show()





# Now do closing
closed = ndimage.binary_dilation(thresholded, structure=np.ones((5, 15)))
closed = ndimage.binary_erosion(closed, structure=np.ones((7, 7)))
closed = ndimage.binary_fill_holes(closed)


plt.figure(figsize = (12, 8)) # Set up the figure to display the results
plt.subplot(1, 2, 1) # Display the thresholded backsub image
plt.imshow(thresholded, cmap='gray')
plt.title("Thresholded Backsub Image")
plt.axis("off")

plt.subplot(1, 2, 2) # Display the image produced from closing
plt.imshow(closed, cmap='gray')
plt.title("Thresholded Image After Closing")
plt.axis("off")

plt.show() # Display the images together




labels = label(closed) # Get the labels from the thresholded image resulting from morphology
regions = regionprops(labels) # Get the properties of the extracted regions

print("Number of regions found: ", len(regions)) # Initially finds 10 regions, try to reduce

validRegions = [] # Keep track of valid regions
extractedRegionsImg = np.zeros_like(labels) # Initialize an image that will show the extracted components
for region in regions: # Loop through all regions
    if region.area > 1500: # Only append regions that are large enough
        validRegions.append(region)
        extractedRegionsImg[labels == region.label] = 1 # Make the pixels for the extracted region white

print("Updated number of regions after eliminating small regions: ", len(validRegions))

plt.figure(figsize=(10, 10)) 
plt.subplot(1, 2, 1) # Display the original thresholded image after closing
plt.imshow(closed, cmap='gray')
plt.title("Thresholded Image After Closing")
plt.axis("off")

plt.subplot(1, 2, 2) # Display the image after extracting the major regions
plt.imshow(extractedRegionsImg, cmap='gray')
plt.title("Extracted Regions from Connected Components")
plt.axis("off")

plt.show()