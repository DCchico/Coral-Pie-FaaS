from adaptive_hist import adaptive_hist
import cv2
import os
import sys


def coldstart(tgraph):
    try:
        # feature extraction:
        image = cv2.imread("function/coldstart.jpeg")
        hist = adaptive_hist(image)

        # vertex and edge add
        v1 = tgraph.addDetection("veh1", "cam1", 2000.0, "")
        v2 = tgraph.addDetection("veh1", "cam2", 2000.0, "")
        tgraph.linkDetection(v1, v2, "")
    except Exception as e:
        print(f"Error in adaptive_hist: {str(e)}", file=sys.stderr)
        # Get the current working directory
        current_directory = os.getcwd()
        print("Current Working Directory:", current_directory, file=sys.stderr)

