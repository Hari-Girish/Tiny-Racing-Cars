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


speedImgFolder = "SpeedRace" # Store speed images folder

speedImgs = [] # Initialize array to hold the images

for i in range(1, 20): # Loop through all action images in the speed race
    file = f"SPEED_{i:02d}.jpeg" # Get the current image
    path = os.path.join(speedImgFolder, file) # Make the file path string
    image = io.imread(path).astype('float64') / 255.0 # Read the image
    image = np.rot90(image, -1) # Rotate image to ensure correct orientation
    image = image[:, :-100, :] # Crop the rightmost part of image out to eliminate the edge of the poster
    speedImgs.append(image) # Add the image to the array of images



# Initialize array to hold the background speed race images
backgroundSpeedImgs = []

for i in range(1, 6): # Loop through all the background speed images
    file = f"SPEED_BG{i:02d}.jpeg" # Get the background image
    path = os.path.join(speedImgFolder, file) # Make the file path string
    image = io.imread(path).astype('float64') / 255.0 # Read the image
    image = np.rot90(image, -1) # Rotate image to ensure correct orientation
    image = image[:, :-100, :] # Crop the rightmost part of image out to eliminate the edge of the poster
    backgroundSpeedImgs.append(image) # Add the image to the array of images



# Display some images for reference
plt.figure(figsize=(12, 8))

# First race frame
plt.subplot(2, 2, 1)
plt.imshow(speedImgs[0])
plt.title("SPEED_01")
plt.axis("off")

# Middle race frame
plt.subplot(2, 2, 2)
plt.imshow(speedImgs[9])
plt.title("SPEED_10")
plt.axis("off")

# Last race frame
plt.subplot(2, 2, 3)
plt.imshow(speedImgs[18])
plt.title("SPEED_19")
plt.axis("off")

# Background without finish line
plt.subplot(2, 2, 4)
plt.imshow(backgroundSpeedImgs[0])
plt.title("Background")
plt.axis("off")


plt.show() # Display the reference images


# Experiment with background subtraction III in region extraction slides
race = speedImgs[0] # Store first race image
background = backgroundSpeedImgs[0] # Store background image without finish line
difference = np.sqrt( # Perform backsub III
    (race[:,:,0] - background[:,:,0])**2 +
    (race[:,:,1] - background[:,:,1])**2 +
    (race[:,:,2] - background[:,:,2])**2
)

print("Backsub Image Dimensions: ", difference.shape) # Print shape to confirm grayscale


# Display the first race image, the background with finish line, and resulting backsub image
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(speedImgs[0])
plt.title("SPEED_01")
plt.axis("off")

# Background without finish line
plt.subplot(2, 2, 2)
plt.imshow(backgroundSpeedImgs[0])      
plt.title("Background")
plt.axis("off")

# Backsub image result
plt.subplot(2, 2, 3)
plt.imshow(difference, cmap='gray')      
plt.title("Background Subtraction Result")
plt.axis("off")

plt.show()


# Backsub works best with background with no finish line, save that as primary background
primarySpeedBG = backgroundSpeedImgs[0]

thresholds = [0.3, 0.5, 0.7, 0.9] # Experiment with some thresholds

plt.figure(figsize=(12, 8)) # Set up figure

for i, T in enumerate(thresholds): # Loop through the thresholds

    thresholded = difference > T # Perform the thresholding

    plt.subplot(2,2,i+1) # Plot the result
    plt.imshow(thresholded, cmap="gray")
    plt.title(f"Threshold = {T}") # Label which threshold was used
    plt.axis("off")

plt.show() # Show the thresholded results


# Results show that threshold of 0.5 looks to be the best

# Convert the background subtraction and thresholding into a function to be used later
def backSubThreshold(image, background, threshold):
    difference = np.sqrt( # Perform backsub III, using each color band
    (image[:,:,0] - background[:,:,0])**2 +
    (image[:,:,1] - background[:,:,1])**2 +
    (image[:,:,2] - background[:,:,2])**2
    )

    thresholded = difference > threshold # Threshold the backsub image

    return thresholded, difference # Return the thresholded backsub image and the original backsub image




# Next experiment with morphology
thresholded, difference = backSubThreshold(speedImgs[0], primarySpeedBG, 0.5) # Call the new backSubThreshold() function to obtain the thresholded image and unthresholded image

# Perform closing, which is dilation followed by erosion
closed = ndimage.binary_dilation(thresholded, structure=np.ones((7, 7))) # Perform the dilation
closed = ndimage.binary_erosion(closed, structure=np.ones((7, 7))) # Perform the erosion
closed = ndimage.binary_fill_holes(closed) # The closing still shows some holes in the cars, lets fill them


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




# Next, experiment with connected components
from skimage.measure import label, regionprops, regionprops_table # Using imports from Homework 3
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

plt.show() # Display the image with the connected components


# Now, work on extracting info from the regions
extractedCars = []
finishLine = None
for region in validRegions: # Loop through the detected regions
    min_y, min_x, max_y, max_x = region.bbox # Store the bounding box of the region
    boxWidth = max_x - min_x # Store the width of the box, used to detect for finishline
    
    if boxWidth > closed.shape[1] * 0.8: # If the width of the component is takes up most of width of image, it must include the finish line
        finishLine = region # We have the finish line
    else: # Else, we must have an individual car
        centroid_y, centroid_x = region.centroid # Store the centroids of the region
        carPixels = (labels == region.label) # Store the pixels of the car
        avgRed = np.mean(race[:, :, 0][carPixels]) # Get the average red color for the car
        avgGreen = np.mean(race[:, :, 1][carPixels]) # Get the average green color for the car
        avgBlue = np.mean(race[:, :, 2][carPixels]) # Get the average blue color for the car

        extractedCars.append({"centroid (y,x)": (centroid_y, centroid_x), "bounding box (min y, min x, max y, max x)": (min_y, min_x, max_y, max_x), "area": region.area, "color": (avgRed, avgGreen, avgBlue), "finished": False}) # Add this car's features to the list of features for this frame

    

# If we have only detected one unique car, that must mean the other car is touching the finish line, has won the race
if (len(extractedCars) < 2 and finishLine is not None): 
    print("Winner Detected")
    extractedCars.append({"centroid (y,x)": "use previous frame", "bounding box (min y, min x, max y, max x)": "merged with finish line", "area": "merged with finish line", "color": "merged with finish line", "finished": True}) # Add features for the winning car

print(extractedCars)



# Write a function to perform the closing and connected components
def closingAndConnectedComponents(image, thresholdedImg):
    # Perform closing, which is dilation followed by erosion
    closed = ndimage.binary_dilation(thresholdedImg, structure=np.ones((7, 7))) # Perform the dilation
    closed = ndimage.binary_erosion(closed, structure=np.ones((7, 7))) # Perform the erosion
    closed = ndimage.binary_fill_holes(closed) # The closing still shows some holes in the cars, lets fill them

    # Temporary displays for testing
    plt.figure(figsize = (12, 8)) # Set up the figure to display the results
    plt.subplot(1, 2, 1) # Display the thresholded backsub image
    plt.imshow(thresholdedImg, cmap='gray')
    plt.title("Thresholded Backsub Image")
    plt.axis("off")

    plt.subplot(1, 2, 2) # Display the image produced from closing
    plt.imshow(closed, cmap='gray')
    plt.title("Thresholded Image After Closing")
    plt.axis("off")

    plt.show() # Display the images together

    from skimage.measure import label, regionprops, regionprops_table # Using imports from Homework 3
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

    plt.show() # Display the image with the connected components

    extractedCars = []
    finishLine = None
    for region in validRegions: # Loop through the detected regions
        min_y, min_x, max_y, max_x = region.bbox # Store the bounding box of the region
        boxWidth = max_x - min_x # Store the width of the box, used to detect for finishline
        
        if boxWidth > closed.shape[1] * 0.8: # If the width of the component is takes up most of width of image, it must include the finish line
            finishLine = region # We have the finish line
        else: # Else, we must have an individual car
            centroid_y, centroid_x = region.centroid # Store the centroids of the region
            carPixels = (labels == region.label) # Store the pixels of the car
            avgRed = np.mean(image[:, :, 0][carPixels]) # Get the average red color for the car
            avgGreen = np.mean(image[:, :, 1][carPixels]) # Get the average green color for the car
            avgBlue = np.mean(image[:, :, 2][carPixels]) # Get the average blue color for the car

            extractedCars.append({"centroid (y,x)": (centroid_y, centroid_x), "bounding box (min y, min x, max y, max x)": (min_y, min_x, max_y, max_x), "area": region.area, "color": (avgRed, avgGreen, avgBlue), "finished": False}) # Add this car's features to the list of features for this frame

        

    # If we have only detected one unique car, that must mean the other car is touching the finish line, has won the race
    if (len(extractedCars) < 2 and finishLine is not None): 
        print("Winner Detected")
        extractedCars.append({"centroid (y,x)": "use previous frame", "bounding box (min y, min x, max y, max x)": "merged with finish line", "area": "merged with finish line", "color": "merged with finish line", "finished": True}) # Add features for the winning car

    print(extractedCars)


# Testing with an image where a car begins crossing the line
thresholded2, difference2 = backSubThreshold(speedImgs[14], primarySpeedBG, 0.5)
closingAndConnectedComponents(speedImgs[14],thresholded2) # Detects 2 major regions instead of 3
