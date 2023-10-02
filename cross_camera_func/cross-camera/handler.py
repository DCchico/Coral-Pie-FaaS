import json
import logging
from event_func import *
from trajectoryGraph import TrajectoryGraph
from coldstart import coldstart
from candidatePool import CandidatePool
from minIO_interface import *
import time
import os
import jsonpickle
import base64
from camera_topo import *

# Initialize logging
logging.basicConfig(level=logging.INFO)

def handle(event, context):
    start_time = time.time()  # Record the start time

    # Initialize OpenFaaS function
    tgraph = TrajectoryGraph()
    pool = CandidatePool()
    coldstart(tgraph) # TODO: Comment this out for real measurement
    try:
        # Deserialize the input data using jsonpickle 
        input_data = jsonpickle.decode(event.body.decode('utf-8'))
        leaving_vehicles = input_data["leaving_vehicles"]
        entering_vehicles = input_data["entering_vehicles"]
        camera_id = input_data["camera_id"]

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

        # Process leaving vehicles
        camera_connections = load_camera_connections('camera_connections.csv')
        reachable_cameras = find_reachable_cameras(camera_connections, camera_id)
        for neighbor_camera in reachable_cameras:
            start_download_time = time.time()
            retrieved_cp = download_candidate_pool_object(minio_client, "candidate-pool", neighbor_camera)
            end_download_time = time.time()
            download_time = end_download_time - start_download_time
            logging.info("Download Time: %.6f seconds", download_time)

            if retrieved_cp != None:
                logging.info("Retrieved Candidate Pool from MinIO Bucket %s: ", neighbor_camera)
                pool = retrieved_cp
            for vertex_info in leaving_vehicles:
                event = {'vertexid': vertex_info["vertexid"],
                    'vehid': vertex_info["vehicle_id"],
                    'camera': camera_id,
                    'timestamp': time.time(),
                    'hist': np.asarray(vertex_info["hist"])}
                json_event = json.dumps(event, cls=NumpyEncoder)
                pool.push(json_event)
            start_upload_time = time.time()
            upload_candidate_pool_object(minio_client, pool, "candidate-pool", neighbor_camera)
            end_upload_time = time.time()
            upload_time = end_upload_time - start_upload_time
            logging.info("Upload Time: %.6f seconds", upload_time)

        start_download_time = time.time()
        retrieved_cp = download_candidate_pool_object(minio_client, "candidate-pool", camera_id)
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        logging.info("Download Time: %.6f seconds", download_time)
        if retrieved_cp != None:
            logging.info("Retrieved Candidate Pool from MinIO Bucket %s: ", camera_id)
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
            "body": "Success"
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
    