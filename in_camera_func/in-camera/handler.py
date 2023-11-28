import cv2
import json
import logging
from event_func import *
from trajectoryGraph import TrajectoryGraph
from coldstart import coldstart
from minIO_interface import *
import time
import os
import jsonpickle
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)

def handle(event, context):
    start_time = time.time()  # Record the start time

    # Initialize OpenFaaS function
    tgraph = TrajectoryGraph()
    vt = VehicleTracking()
    coldstart(tgraph)
    try:
        # Process input data from event (assuming JSON data)
        input_data = json.loads(event.body.decode('utf-8'))
        camera_id = input_data.get('camera_id')
        frame_id = int(input_data.get('frame_id'))
        encoded_image = input_data.get('image')
        bboxes = input_data.get('bboxes')

        # Initialize MinIO client
        minio_client = Minio(
            endpoint=os.environ.get("MINIO_ADDR"), 
            access_key=os.environ.get("MINIO_ACCESS_KEY"), 
            secret_key=os.environ.get("MINIO_SECRET_KEY"), 
            secure=False,
        )
        # minio_client = Minio(
        #     endpoint="localhost:9002", 
        #     access_key="WDuYITpMKOJSIkWqkYXb", 
        #     secret_key="4hHAS2NLKqg2vnk06qzho1DjTODdQCVrZkJTy0Lb", 
        #     secure=False,
        # )
        # Retrieve the VehicleTracking object
        start_download_time = time.time()
        retrieved_vt = download_vehicle_tracking_object(minio_client, "vehicle-tracking", camera_id)
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        logging.info("Download Time: %.6f seconds", download_time)

        if retrieved_vt != None:
            logging.info("Retrieved Tracklets from MinIO Bucket %s: ", camera_id)
            logging.info("Length of Tracklets is %s\n", len(retrieved_vt.tracklets))
            vt = retrieved_vt

        # Load image and perform processing
        # Decode base64-encoded image data
        decoded_image = base64.b64decode(encoded_image)
        # Convert decoded data to numpy array
        nparr = np.frombuffer(decoded_image, np.uint8)
        # Read the image using cv2.imdecode
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logging.info("Frame id: %s", frame_id)

        tracked_bboxes = vt.sort_update(bboxes)
        logging.info("Updated Vehicle Tracklets with new Trackers number: %d", len(tracked_bboxes))

        vehicle_of_interest = vt.status_update(tgraph, camera_id, frame_id, image)
        leaving_vehicles = vehicle_of_interest["leaving_vehicle"]
        entering_vehicles = vehicle_of_interest["entering_vehicles"]
        logging.info("Extracted %d Vehicles Leaving with age > 3", len(leaving_vehicles))

        response_data = {
            "leaving_vehicles": [],
            "entering_vehicles": [],
            "camera_id": camera_id,
        }  # Lists to store [vertexid, vehicle.id, hist]

        for vehicle in leaving_vehicles:
            logging.info("Vehicle ID: %d is leaving" % vehicle.id)
            hist = feature_extraction_adaptive_histogram(vehicle)
            vertexid = vehicle.vertexid
            # Serialize the relevant information from the Vertex object
            vertex_info = {
                "vertexid": vertexid,
                "vehicle_id": vehicle.id,
                "hist": hist
            }
            response_data["leaving_vehicles"].append(vertex_info)
        
        for vehicle in entering_vehicles:
            logging.info("Vehicle ID: %d is leaving" % vehicle.id)
            hist = feature_extraction_adaptive_histogram(vehicle)
            vertexid = vehicle.vertexid
            # Serialize the relevant information from the Vertex object
            vertex_info = {
                "vertexid": vertexid,
                "vehicle_id": vehicle.id,
                "hist": hist
            }
            response_data["entering_vehicles"].append(vertex_info)
        

        # To upload the VehicleTracking object
        start_upload_time = time.time()
        upload_vehicle_tracking_object(minio_client, vt, "vehicle-tracking", camera_id)
        end_upload_time = time.time()
        upload_time = end_upload_time - start_upload_time
        logging.info("Upload Time: %.6f seconds", upload_time)

        tgraph.shutdown()
        logging.info("Finished Iteration")

        end_time = time.time()  # Record the end time
        latency = end_time - start_time  # Calculate the latency in seconds
    
        logging.info("End-to-end latency: %.6f seconds", latency)

        response_data["download_time"] = download_time
        response_data["upload_time"] = upload_time
        # Return a response with the list of lists
        response = {
            "statusCode": 200,
            "body": jsonpickle.encode(response_data)
        }
        return response

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        for key in input_data.keys():
            logging.error("Keys in input_data: %s", key)
        response = {
            "statusCode": 500,
            "body": "Error occurred: " + str(e)
        }
        return response
    
if __name__ == "__main__":
    with open('coldstart.jpeg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    for i in range(5):
        input_data = {
            'body': json.dumps({
                "camera_id": "0",
                "frame_id": str(i),
                "image": encoded_image,  # Replace with actual image data
                "bboxes": [[183.3546, 233.2469, 267.5851, 296.2590, 100]] if i == 0 else []
            })
        }

        result = handle(input_data, None)
        print("--------------------------\n", result)