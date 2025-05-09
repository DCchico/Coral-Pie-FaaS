import json
import logging
from event_func import *
from trajectoryGraph import TrajectoryGraph
from coldstart import coldstart
from candidatePool import CandidatePool
from redis_interface import *
import time
import os
import jsonpickle
import base64
from camera_topo import *

# Initialize logging
logging.basicConfig(level=logging.INFO)

redis_host = os.environ.get("REDIS_ADDR", "128.110.217.239")
# redis_host = "128.110.217.193"
redis_port = 6379
redis_password = None  # Add password if your Redis requires authentication

# Initialize Redis client
redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    socket_timeout=5,
    decode_responses=False  # We need binary data for pickled objects
)
# Initialize OpenFaaS function
tgraph = TrajectoryGraph()
pool = CandidatePool()
coldstart(tgraph)


def handle(event, context):
    global tgraph, pool
    start_time = time.time()  # Record the start time
    download_time = 0
    upload_time = 0
    latency = 0
    try:
        # Deserialize the input data using jsonpickle 
        input_data = jsonpickle.decode(event.body)
        leaving_vehicles = input_data["leaving_vehicles"]
        entering_vehicles = input_data["entering_vehicles"]
        camera_id = input_data["camera_id"]

        # Process leaving vehicles
        camera_connections = load_camera_connections('/home/app/function/camera_connections.csv')
        # Debugging path:
        # camera_connections = load_camera_connections('camera_connections.csv')
        reachable_cameras = find_reachable_cameras(camera_connections, camera_id)

        for neighbor_camera in reachable_cameras:
            start_download_time = time.time()
            retrieved_cp = download_candidate_pool_object(redis_client, "candidate-pool", neighbor_camera)
            end_download_time = time.time()
            download_time = end_download_time - start_download_time
            logging.info("Download Time: %.6f seconds", download_time)

            if retrieved_cp != None:
                logging.info("Retrieved Candidate Pool from Redis %s: ", neighbor_camera)
                pool = retrieved_cp
            for vertex_info in leaving_vehicles:
                event = {'vertexid': vertex_info["vertexid"],
                    'vehid': vertex_info["vehicle_id"],
                    'camera': camera_id,
                    'timestamp': time.time(),
                    'hist': np.asarray(vertex_info["hist"])}
                # json_event = json.dumps(event, cls=NumpyEncoder)
                pool.push(event)
            start_upload_time = time.time()
            upload_candidate_pool_object(redis_client, pool, "candidate-pool", neighbor_camera)
            end_upload_time = time.time()
            upload_time = end_upload_time - start_upload_time
            logging.info("Upload Time: %.6f seconds", upload_time)

        start_download_time = time.time()
        retrieved_cp = download_candidate_pool_object(redis_client, "candidate-pool", camera_id)
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        logging.info("Download Time: %.6f seconds", download_time)
        if retrieved_cp != None:
            logging.info("Retrieved Candidate Pool from Redis %s: ", camera_id)
            pool = retrieved_cp
        for vertex_info in entering_vehicles:
            vertexid = vertex_info["vertexid"]
            res = pool.matching(vertex_info["hist"], 0.1)
            edge_storage(tgraph, res, vertexid)
      
        tgraph.shutdown()
        logging.info("Finished Iteration")

        end_time = time.time()  # Record the end time
        latency = end_time - start_time  # Calculate the latency in seconds
    
        logging.info("End-to-end latency: %.6f seconds", latency)


        # Return a response with the list of lists
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "download_time": download_time * 1000.0,
                "upload_time": upload_time * 1000.0,
                "latency": latency * 1000.0
            }),
            "headers": {
                "Content-Type": "application/json"
            }
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
    # Create a mock event object (similar to what OpenFaaS would send)
    class MockEvent:
        def __init__(self, body_data):
            self.body = body_data

    # Create test data
    test_data = {
        "camera_id": "camera_1",
        "leaving_vehicles": [
            {
                "vertexid": "vertex1",
                "vehicle_id": "veh1",
                "hist": [0.1, 0.2, 0.3]  # Mock histogram
            }
        ],
        "entering_vehicles": [
            {
                "vertexid": "vertex2",
                "vehicle_id": "veh2",
                "hist": [0.2, 0.3, 0.4]  # Mock histogram
            }
        ]
    }

    # Create mock event with encoded data
    mock_event = MockEvent(jsonpickle.encode(test_data))

    # Test the handler
    print("\n==== Testing Cross-Camera Handler ====")
    result = handle(mock_event, None)
    print("==== Result ====")
    print(json.dumps(result, indent=2))
    print("====================\n")
    