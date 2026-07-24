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


distImgFolder = "DistanceRaceNew" # Store the folder

distImgs = [] # Initialize storage for the images


for i in range(1, 8): # Loop through all action images in the speed race
    file = f"DIST_NEW_{i:02d}.jpeg" # Get the current image
    path = os.path.join(distImgFolder, file) # Make the file path string
    image = io.imread(path).astype('float64') / 255.0 # Read the image
    image = np.rot90(image, -1) # Rotate image to ensure correct orientation
    image = image[300:, :, :] # Crop the top part of image out to eliminate the wall
    #image = filters.gaussian(image, sigma=2.0, truncate=3, channel_axis=1)
    distImgs.append(image) # Add the image to the array of images



file = "DIST_NEW_BG_01.jpeg" # Get the background image
path = os.path.join(distImgFolder, file) # Make the file path string
image = io.imread(path).astype('float64') / 255.0 # Read the image
image = np.rot90(image, -1) # Rotate image to ensure correct orientation
image = image[300:, :, :] # Crop the top part of image out to eliminate the wall
#image = filters.gaussian(image, sigma=2.0, truncate=3, channel_axis=1)
background = image # Store our background image



# Display some images for reference
plt.figure(figsize=(12, 8))

# First race frame
plt.subplot(2, 2, 1)
plt.imshow(distImgs[0])
plt.title("DIST_NEW_01")
plt.axis("off")

# Middle race frame
plt.subplot(2, 2, 2)
plt.imshow(distImgs[3])
plt.title("DIST_NEW_04")
plt.axis("off")

# Last race frame
plt.subplot(2, 2, 3)
plt.imshow(distImgs[6])
plt.title("DIST_NEW_07")
plt.axis("off")

# Background
plt.subplot(2, 2, 4)
plt.imshow(background)
plt.title("Background")
plt.axis("off")


plt.show() # Display the reference images





# Experiment with background subtraction III in region extraction slides
race = distImgs[3] # Store first race image 
difference = np.sqrt( # Perform backsub III
    (race[:,:,0] - background[:,:,0])**2 +
    (race[:,:,1] - background[:,:,1])**2 +
    (race[:,:,2] - background[:,:,2])**2
)

print("Backsub Image Dimensions: ", difference.shape) # Print shape to confirm grayscale

thresholded = difference > 0.6 # Threshold the backsub image


# Display a race image
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(distImgs[3])
plt.title("DIST_NEW_04")
plt.axis("off")

# Background
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
plt.title("Threshold = 0.6")
plt.axis("off")

plt.show()



# Now do closing
closed = ndimage.binary_dilation(thresholded, structure=np.ones((7, 7)))
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