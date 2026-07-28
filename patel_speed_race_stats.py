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

from patel_speed_race import runSpeedRace



raceHistory = runSpeedRace() # Run the speed race to gather the full race history


def speedRaceStats(raceHistory): # Function to get the total race stats
    redCarHistory = [frame[0] for frame in raceHistory] # Acquire the red car feature history
    blueCarHistory = [frame[1] for frame in raceHistory] # Acquire the blue car feature history

    redStats = carStats(redCarHistory, "Red Car") # Get the red car stats
    blueStats = carStats(blueCarHistory, "Blue Car") # Get the blue car stats

    lines = [] # All output lines will go in here to output to a file
    lines.append("=" * 70)
    lines.append("Speed Race Statistics")
    lines.append("=" * 70)
    lines.append(f"Total Frames Analyzed: {len(raceHistory)}")
    lines.append("\n\n\n")

    for stats in [redStats, blueStats]: # Loop through the two sets of stats
        lines.append(f"==== {stats['name']}============================") # Get the name, red car or blue car
        lines.append(f"* Total Frames                                   :   {stats['total frames']}") # Get the number of frames
        lines.append(f"* Change in X (Pixels)                           :   {stats['change in x']}") # Change in x position
        lines.append(f"* Change in Y (Pixels)                           :   {stats['change in y']}") # Change in y pixels
        lines.append(f"* Total Distance Traveled (Pixels)               :   {stats['total distance traveled']}") # Total distance traveled
        lines.append(f"* Approximate Pixels per Frame                   :   {stats['approximate pixels per frame']}") # Pixels per frame
        lines.append(f"* Approximate Pixels per Second                  :   {stats['approximate pixels per second']}") # Pixels per second
        lines.append(f"* Y Pixels per Frame                             :   {stats['y pixels per frame']}") # Y pixels per frame
        lines.append(f"* Y Pixels per Second                            :   {stats['y pixels per second']}") # Y pixels per second
        lines.append(f"* Maximum Distance Traveled Between Two Frames   :   {stats['maximum distance traveled between two frames']}") # Maximum distance between two frames
        lines.append(f"* Average Area of Car                            :   {stats['average area']}") # Average area
        lines.append(f"* Standard Deviation of Area                     :   {stats['standard deviation of area']}") # Standard deviation of area
        lines.append("\n\n")
    
    

    # Print the detected winner
    lastFrame = raceHistory[-1] # Store the last frame
    redCar = lastFrame[0] # Store the red car features
    blueCar = lastFrame[1] # Store the blue car features
    redFinished = redCar["finished"] # Store the boolean value indicating if the red car finished the race
    blueFinished = blueCar["finished"] # Store the boolean value indicating if the blue car finished the race
    
    if redFinished: # If the red car won, print its victory message
        lines.append("Winner Detected! The red car wins!")
    elif blueFinished: # If the blue car won, print its victory message
        lines.append("Winner Detected! The blue car wins!")

    
    with open("patel_speed_race_stats_results.txt", "w") as f:
        f.write("\n".join(lines)) # Write to the output file




def carStats(carHistory, carName, fps = 30): # Function to get individual stats for one car
    numberFrames = len(carHistory) # Get the number of frames
    

    # Last frame for winner car has string placeholders for features, use second to last frame's stats in that case
    yList = []
    xList = []
    areaList = []

    for car in carHistory: # Loop through all frames for this car
        centroid = car["centroid (y,x)"] # Store the centroids

        if centroid == "use previous frame": # Placeholder for winning car's last frame
            yList.append(yList[-1]) # Use last saved y coord
            xList.append(xList[-1]) # Use last saved x coord
        else:
            yList.append(float(centroid[0])) # Else, save the numercial value
            xList.append(float(centroid[1])) # Else, save the numerical value

        # Handle the area now
        area = car["area"] # Get the area value
        if area == "merged with finish line": # Placeholder for winning car's last frame
            areaList.append(areaList[-1]) # Reuse last saved area
        else: 
            areaList.append(float(area)) # Else, save the numerical value


    # Extract all centroids
    yCoords = np.array(yList, dtype=float) # Get all the y coordinates of the centroids
    xCoords = np.array(xList, dtype=float) # Get all the x coordinates of the centroids
    
    areas = np.array(areaList, dtype=float) # Get all areas for the car

    changeY = yCoords[0] - yCoords[-1] # Get the change in y, smaller y values as car goes up
    changeX = xCoords[-1] - xCoords[0] # Get the change in x

    # Acquire Euclidean distances
    ySub = np.diff(yCoords) # Subtract the y's
    xSub = np.diff(xCoords) # Subtract the x's
    distances = np.sqrt(xSub**2 + ySub**2) # Calculate Euclidean distances
    totalDistance = np.sum(distances) # Calculate total sum of all Euclidean distances


    # Estimate total speed
    pixelsPerFrame = totalDistance / numberFrames # Calculate pixels traveled per frame
    pixelsPerSecond = pixelsPerFrame * fps # Approximate real time speed based on choice of fps

    # Estimate speed only in y direction since the cars mainly move up
    yPixelsPerFrame = changeY / numberFrames # Y pixels traveled per frame
    yPixelsPerSecond = yPixelsPerFrame * fps # Real time vertical speed


    # See how consistent the area of the car is
    avgArea = np.mean(areas) # Average area
    stdArea = np.std(areas) # Standard deviation of areas

    return {
        "name": carName,
        "total frames": numberFrames,
        "change in x": changeX,
        "change in y": changeY,
        "total distance traveled": totalDistance,
        "approximate pixels per frame": pixelsPerFrame,
        "approximate pixels per second": pixelsPerSecond,
        "y pixels per frame": yPixelsPerFrame,
        "y pixels per second": yPixelsPerSecond,
        "maximum distance traveled between two frames": np.max(distances),
        "average area": avgArea,
        "standard deviation of area": stdArea
    }




speedRaceStats(raceHistory) # Call the function to gather the full race stats
