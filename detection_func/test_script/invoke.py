import requests
import json

# Define the URL of your deployed OpenFaaS function
function_url = "http://127.0.0.1:31112/function/detection"

# Load an image as bytes (replace with your image path)
with open('coldstart.jpeg', 'rb') as image_file:
    image_data = image_file.read()

# Send a POST request to your function
response = requests.post(function_url, data=image_data, headers={"Content-Type": "application/json"})

# Check the response status code
if response.status_code == 200:
    # Parse the JSON response
    result = json.loads(response.text)
    bounding_boxes = result["bounding_boxes"]
    preprocess_time = result["preprocessing_time"]
    prediction_time = result["prediction_time"]
    
    # Process the bounding boxes or other results as needed
    print("Bounding Boxes:", bounding_boxes)
    print("Preprocessing Time:", preprocess_time)
    print("Prediction Time:", prediction_time)
else:
    print("Error:", response.status_code, response.text)
