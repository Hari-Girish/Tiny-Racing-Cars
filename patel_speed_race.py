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








# Convert the background subtraction and thresholding into a function to be used later
def backSubThreshold(image, background, threshold):
    difference = np.sqrt( # Perform backsub III, using each color band
    (image[:,:,0] - background[:,:,0])**2 +
    (image[:,:,1] - background[:,:,1])**2 +
    (image[:,:,2] - background[:,:,2])**2
    )

    thresholded = difference > threshold # Threshold the backsub image

    return thresholded, difference # Return the thresholded backsub image and the original backsub image





# Write a function to perform the closing and connected components
def closingAndConnectedComponents(image, thresholdedImg):
    # Perform closing, which is dilation followed by erosion
    closed = ndimage.binary_dilation(thresholdedImg, structure=np.ones((7, 7))) # Perform the dilation
    closed = ndimage.binary_erosion(closed, structure=np.ones((7, 7))) # Perform the erosion
    closed = ndimage.binary_fill_holes(closed) # The closing still shows some holes in the cars, lets fill them
    """
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
    """

    
    labels = label(closed) # Get the labels from the thresholded image resulting from morphology
    regions = regionprops(labels) # Get the properties of the extracted regions
    #print("Number of regions found: ", len(regions)) # Initially finds 10 regions, try to reduce

    validRegions = [] # Keep track of valid regions
    extractedRegionsImg = np.zeros_like(labels) # Initialize an image that will show the extracted components
    for region in regions: # Loop through all regions
        if region.area > 1500: # Only append regions that are large enough
            validRegions.append(region)
            extractedRegionsImg[labels == region.label] = 1 # Make the pixels for the extracted region white

    #print("Updated number of regions after eliminating small regions: ", len(validRegions))
    """
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
    """
    extractedCars = []
    finishLine = None
    for region in validRegions: # Loop through the detected regions
        min_y, min_x, max_y, max_x = region.bbox # Store the bounding box of the region
        boxWidth = max_x - min_x # Store the width of the box, used to detect for finishline
        
        if boxWidth > closed.shape[1] * 0.8: # If the width of the component takes up most of width of image, it must include the finish line
            finishLine = region # We have the finish line
        else: # Else, we must have an individual car
            centroid_y, centroid_x = region.centroid # Store the centroids of the region
            carPixels = (labels == region.label) # Store the pixels of the car
            avgRed = np.mean(image[:, :, 0][carPixels]) # Get the average red color for the car
            avgGreen = np.mean(image[:, :, 1][carPixels]) # Get the average green color for the car
            avgBlue = np.mean(image[:, :, 2][carPixels]) # Get the average blue color for the car

            extractedCars.append({"centroid (y,x)": (centroid_y, centroid_x), "bounding box (min y, min x, max y, max x)": (min_y, min_x, max_y, max_x), "area": region.area, "color": (avgRed, avgGreen, avgBlue), "finished": False}) # Add this car's features to the list of features for this frame

        
    winnerDetected = False # Initialize a boolean to indicate when a winner has been detected

    # If we have only detected one unique car, that must mean the other car is touching the finish line, has won the race
    if (len(extractedCars) < 2 and finishLine is not None): 
        #print("Winner Detected")
        extractedCars.append({"centroid (y,x)": "use previous frame", "bounding box (min y, min x, max y, max x)": "merged with finish line", "area": "merged with finish line", "color": "merged with finish line", "finished": True}) # Add features for the winning car
        winnerDetected = True

    #print(extractedCars)
    extractedCars.sort(key = lambda car: car["color"][0] if isinstance(car["color"], tuple) else -1, reverse = True) # Sort so the red car features are before the blue car features, will also apply when the blue car wins

    return extractedCars, winnerDetected # Return the car features for this frame




# Begin setting up the loop to go through all the speed race images until we detect a winner

raceHistory = [] # Initialize an array to hold the car features extracted at each frame

# Backsub works best with background with no finish line, save that as primary background
primarySpeedBG = backgroundSpeedImgs[0]

threshold = 0.5 # The best threshold for background subtraction was found to be 0.5


for i in range(1, 20): # Begin the loop to go through all speed race images. Loop breaks if winner detected before last frame
    currImg = speedImgs[i - 1] # Store the current speed race image

    thresholded, difference = backSubThreshold(currImg, primarySpeedBG, threshold) # Call the function to perform the background subtraction

    currFrameFeatures, isWinner = closingAndConnectedComponents(currImg, thresholded) # Perform the closing and connected components, extract the car features

    raceHistory.append(currFrameFeatures) # Add the extracted car features to the race history of car features

    if isWinner: # If we have detected a winner, break from the loop, we do not need to analyze more frames
        # Write code to print the winner
        lastFrame = raceHistory[-1] # Store the last frame
        redCar = lastFrame[0] # Store the red car features
        blueCar = lastFrame[1] # Store the blue car features
        redFinished = redCar["finished"] # Store the boolean value indicating if the red car finished the race
        blueFinished = blueCar["finished"] # Store the boolean value indicating if the blue car finished the race
        
        if redFinished: # If the red car won, print its victory message
            print("Winner Detected! The red car wins!")
        elif blueFinished: # If the blue car won, print its victory message
            print("Winner Detected! The blue car wins!")

        break # The winner has been declared, we do not need to process any more images
    




