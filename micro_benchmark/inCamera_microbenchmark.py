import requests
import json
import time
import os
import jsonpickle
import base64
import statistics
import ast

# Define the URL of your deployed OpenFaaS function
inCamerafunction_url = "http://127.0.0.1:31112/function/in-camera"
crossCamerafunction_url = "http://127.0.0.1:31112/function/cross-camera"

# Prepare input BBoxes
bbox_InputList = []
with open("bbox.text", 'r') as bbox_file:
    for line in bbox_file:
        if line.startswith("Bounding Boxes:"):
            bboxString = line[line.find("["):]
            bboxList = ast.literal_eval(bboxString)
            bbox_InputList.append(bboxList)


e2e_inCamera_time = []
inCamera_download_time = []
inCamera_upload_time = []
e2e_crossCamera_time = []
crossCamera_download_time = []
crossCamera_upload_time = []
# Load an image as bytes and prepare input data structure
image_path = "../camera_stream_data/"
for i in range(100):
    image_name = os.path.join(image_path, f"{i:06d}.jpeg")
    with open(image_name, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    input_data = {
                "camera_id": "camera_0",
                "frame_id": str(i),
                "image": image_data,  
                "bboxes": bbox_InputList[i]
            }

    # Send a POST request to inCamera function
    start_inCamera = time.time()
    response = requests.post(inCamerafunction_url, json=input_data, headers={"Content-Type": "application/json"})
    end_inCamera = time.time()
    e2e_inCamera = (end_inCamera - start_inCamera) * 1000
    e2e_inCamera_time.append(e2e_inCamera)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        result = jsonpickle.decode(response.content)
        inCamera_download_time.append(result["download_time"] * 1000.0)
        inCamera_upload_time.append(result["upload_time"] * 1000.0)
    else:
        print("In-Camera Function Error:", response.status_code, response.text)

    # Send a POST request to inCamera function
    start_crossCamera = time.time()
    intermediate_result = response.content
    response = requests.post(crossCamerafunction_url, data=intermediate_result, headers={"Content-Type": "application/json"})
    end_crossCamera = time.time()
    e2e_crossCamera = (end_crossCamera - start_crossCamera) * 1000
    e2e_crossCamera_time.append(e2e_crossCamera)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        result = jsonpickle.decode(response.content)
        crossCamera_download_time.append(result["download_time"])
        # crossCamera_upload_time.append(result["upload_time"] * 1000.0)
    else:
        print("Cross-Camera Function Error:", response.status_code, response.text)

# Summerize Metrics
print("In-Camera Download Time:", statistics.mean(inCamera_download_time))
print("In-Camera Upload Time:", statistics.mean(inCamera_upload_time))
print("In-Camera E2E Latency:", statistics.mean(e2e_inCamera_time))
print("Cross-Camera Download Time:", statistics.mean(crossCamera_download_time))
# print("Cross-Camera Upload Time:", statistics.mean(crossCamera_upload_time))
print("Cross-Camera E2E Latency:", statistics.mean(e2e_crossCamera_time))

with open('metrics-InCrossCameraBench.txt', 'w') as file:
    json.dump(inCamera_download_time, file)
    json.dump(inCamera_upload_time, file)
    json.dump(e2e_inCamera_time, file)
    json.dump(crossCamera_download_time, file)
    # json.dump(crossCamera_upload_time, file)
    json.dump(e2e_crossCamera_time, file)