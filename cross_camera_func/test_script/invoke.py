import requests
import json
import base64
import jsonpickle
import time
import statistics
from flask import Flask, request
from minio import Minio


app = Flask(__name__)

resource = {}

@app.route('/webhook', methods=['POST'])
def receive_webhook():
  
    data = request.json  
    api = data["api"]
    print(data)


    return {"message": "Resource updated successfully", "resource": resource}, 200


# Load the image from a file and encode it as base64
image_path = "coldstart.jpeg"
with open('coldstart.jpeg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

url_in_camera = "http://127.0.0.1:31112/function/in-camera"
url_cross_camera = "http://127.0.0.1:31112/function/cross-camera"


input_data = {
                "camera_id": "camera_0",
                "frame_id": "1",
                "image": encoded_image,  # Replace with actual image data
                "bboxes": [[183.3546, 233.2469, 267.5851, 296.2590, 100]]
            }
num_requests = 2
invoke_times = []
for _ in range(num_requests):
    
    response = requests.post(url_in_camera, json=input_data)
    print(response.text)
    intermediate_data = response.content
    start_time = time.time()
    response = requests.post(url_cross_camera, data=input_data)
    end_time = time.time()
    invoke_time = (end_time - start_time) * 1000
    invoke_times.append(invoke_time)

average_invoke_time = statistics.mean(invoke_times)
std_deviation_invoke_time = statistics.stdev(invoke_times)
print("\nAverage Invoke Time: %.3f ms" % average_invoke_time)
print("Standard Deviation of Invoke Time: %.3f ms" % std_deviation_invoke_time)

# print("Invoke Time: %.3f seconds", invoke_time)
# print(response.status_code)
# print(jsonpickle.decode(response.content))
