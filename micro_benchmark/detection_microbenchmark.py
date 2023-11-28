import requests
import json
import time
import os
import statistics

# Define the URL of your deployed OpenFaaS function
function_url = "http://127.0.0.1:31112/function/detection"

preprocess_time = []
prediction_time = []
e2e_detection_time = []
# Load an image as bytes (replace with your image path)
image_path = "../camera_stream_data/"
for i in range(100):
    image_name = os.path.join(image_path, f"{i:06d}.jpeg")
    with open(image_name, 'rb') as image_file:
        image_data = image_file.read()

    # Send a POST request to your function
    start_detection = time.time()
    response = requests.post(function_url, data=image_data, headers={"Content-Type": "application/json"})
    end_detection = time.time()
    e2e_detection = (end_detection - start_detection) * 1000
    e2e_detection_time.append(e2e_detection)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        result = json.loads(response.text)
        bounding_boxes = result["bounding_boxes"]
        loadModel_time = result["model_loading_time"]
        preprocess_time.append(result["preprocessing_time"])
        prediction_time.append(result["prediction_time"])
        
        # Process the bounding boxes or other results as needed
        print("Bounding Boxes:", bounding_boxes)
        print("Model Loading Time:", loadModel_time)
    else:
        print("Error:", response.status_code, response.text)

# Summerize Metrics
print("Preprocessing Time:", statistics.mean(preprocess_time))
print("Prediction Time:", statistics.mean(prediction_time))
print("E2E Latency:", statistics.mean(e2e_detection_time))