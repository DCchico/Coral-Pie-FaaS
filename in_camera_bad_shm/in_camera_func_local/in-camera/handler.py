import cv2
import json
import logging
import traceback
from event_func import *
from trajectoryGraph import TrajectoryGraph
from coldstart import coldstart
import time
import os
import jsonpickle
import base64
import numpy as np
import pickle
import sys

# Initialize logging with clear format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Global variables for initialization
tgraph = None
vt = None
tgraph = TrajectoryGraph()
vt = VehicleTracking()
coldstart(tgraph)

# In-memory storage to replace MinIO
in_memory_storage = {}  # Dictionary to store objects by camera_id

# Function to get size of object in KB
def get_size_kb(obj):
    try:
        # Use pickle to get a more accurate size of complex objects
        return len(pickle.dumps(obj)) / 1024.0
    except:
        # Fallback to sys.getsizeof if pickle fails
        return sys.getsizeof(obj) / 1024.0

def handle(req, context):
    global tgraph, vt
    """
    Handler function that processes vehicle tracking from images
    """
    start_time = time.time()  # Record the overall start time
    metrics = {
        "timings": {},
        "sizes": {},
    }
    
    logging.info("Handler invoked at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Request type: %s", type(req))
    
    try:
        # --- TIMING: Input parsing ---
        parse_start = time.time()
        
        # Extract the request body from different possible formats
        if isinstance(req, dict) and 'Payload' in req:
            # STateFlow format - payload is base64 encoded
            logging.info("STateFlow request format detected")
            payload_base64 = req['Payload']
            try:
                body_str = base64.b64decode(payload_base64).decode('utf-8')
                logging.info("Successfully decoded base64 payload")
            except Exception as e:
                logging.error("Failed to decode base64 payload: %s", str(e))
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Invalid base64 payload: {str(e)}"})
                }
        elif hasattr(req, 'body'):
            # OpenFaaS format
            logging.info("OpenFaaS request format detected")
            body_str = req.body.decode('utf-8') if isinstance(req.body, bytes) else str(req.body)
        elif isinstance(req, bytes):
            # Raw bytes
            logging.info("Raw bytes request format detected")
            body_str = req.decode('utf-8')
        elif isinstance(req, str):
            # Already a string
            logging.info("String request format detected")
            body_str = req
        else:
            # Assume it can be converted to string
            logging.info("Unknown request format, attempting string conversion")
            body_str = str(req)
            
        # Parse JSON
        try:
            input_data = json.loads(body_str)
            logging.info("Successfully parsed JSON input")
        except json.JSONDecodeError as e:
            logging.error("JSON parsing error: %s", str(e))
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid JSON: {str(e)}"})
            }
        
        # --- TIMING: Validation ---
        validation_start = time.time()
        metrics["timings"]["input_parsing"] = validation_start - parse_start
        
        # Extract required fields with validation
        camera_id = input_data.get('camera_id')
        if not camera_id:
            logging.error("Missing camera_id field")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing camera_id field"})
            }
            
        # Safe frame_id conversion
        try:
            frame_id = int(input_data.get('frame_id'))
            logging.info("Frame ID: %d", frame_id)
        except (ValueError, TypeError) as e:
            logging.error("Invalid frame_id: %s", str(e))
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "frame_id must be convertible to an integer"})
            }
            
        encoded_image = input_data.get('image')
        if not encoded_image:
            logging.error("Missing image field")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing image field"})
            }
            
        bboxes = input_data.get('bboxes')
        if not bboxes or not isinstance(bboxes, list):
            logging.error("Missing or invalid bboxes field")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing or invalid bboxes field"})
            }

        # Check for test_mode which skips CV operations
        if input_data.get('test_mode', False):
            logging.info("TEST MODE ACTIVATED - skipping CV operations")
            # Return a simple response without any CV processing
            response_data = {
                "message": "Test mode successful",
                "camera_id": camera_id,
                "leaving_vehicles": [],
                "entering_vehicles": [],
                "metrics": {
                    "input_parsing": metrics["timings"]["input_parsing"],
                    "validation": time.time() - validation_start,
                    "total": time.time() - start_time
                }
            }
            
            response = {
                "statusCode": 200,
                "body": json.dumps(response_data)  # Use regular json for test mode
            }
            return response

        # --- TIMING: Initialization ---
        init_start = time.time()
        metrics["timings"]["validation"] = init_start - validation_start
        
        # Initialize OpenFaaS function
        logging.info("Getting trajectory graph and vehicle tracking objects")
        if tgraph is None or vt is None:
            tgraph = TrajectoryGraph()
            vt = VehicleTracking()
            coldstart(tgraph)
        
        # --- TIMING: In-memory retrieval ---
        retrieve_start = time.time()
        metrics["timings"]["initialization"] = retrieve_start - init_start
        
        # Retrieve the VehicleTracking object from in-memory storage
        start_download_time = time.time()
        retrieved_vt = in_memory_storage.get(camera_id)
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        logging.info("In-memory Retrieval Time: %.6f seconds", download_time)

        # Measure size of retrieved data if it exists
        retrieved_size_kb = 0
        if retrieved_vt is not None:
            retrieved_size_kb = get_size_kb(retrieved_vt)
            logging.info("Retrieved object size: %.2f KB", retrieved_size_kb)
            logging.info("Retrieved Tracklets from in-memory storage for camera %s", camera_id)
            logging.info("Length of Tracklets is %d", len(retrieved_vt.tracklets))
            vt = retrieved_vt

        # --- TIMING: Image decoding ---
        decode_start = time.time()
        metrics["timings"]["retrieval"] = decode_start - retrieve_start
        
        # Decode image with enhanced error handling
        try:
            decoded_image = base64.b64decode(encoded_image)
            nparr = np.frombuffer(decoded_image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            logging.info("Successfully decoded image, shape: %s", str(image.shape))
        except Exception as e:
            logging.error("Error decoding image: %s", str(e))
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Image decoding error: {str(e)}"})
            }

        # --- TIMING: Sort update (tracking) ---
        sort_start = time.time()
        metrics["timings"]["image_decoding"] = sort_start - decode_start
        metrics["sizes"]["image_kb"] = len(decoded_image) / 1024.0
        
        # Process tracking
        try:
            logging.info("Updating vehicle tracking with %d bounding boxes", len(bboxes))
            tracked_bboxes = vt.sort_update(bboxes)
            logging.info("Updated vehicle tracklets, number of trackers: %d", len(tracked_bboxes))

            # --- TIMING: Status update ---
            status_start = time.time()
            metrics["timings"]["sort_update"] = status_start - sort_start
            
            vehicle_of_interest = vt.status_update(tgraph, camera_id, frame_id, image)
            
            # --- TIMING: Feature extraction prep ---
            feature_start = time.time()
            metrics["timings"]["status_update"] = feature_start - status_start
            
            leaving_vehicles = vehicle_of_interest["leaving_vehicle"]
            entering_vehicles = vehicle_of_interest["entering_vehicles"]
            logging.info("Extracted %d vehicles leaving and %d vehicles entering", 
                         len(leaving_vehicles), len(entering_vehicles))
        except Exception as e:
            logging.error("Error in tracking update: %s", str(e))
            logging.error(traceback.format_exc())
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Tracking error: {str(e)}"})
            }

        # Build response
        response_data = {
            "leaving_vehicles": [],
            "entering_vehicles": [],
            "camera_id": camera_id,
        }

        # --- TIMING: Process leaving vehicles ---
        leaving_feature_start = time.time()
        
        # Process leaving vehicles
        for vehicle in leaving_vehicles:
            logging.info("Vehicle ID: %d is leaving", vehicle.id)
            hist = feature_extraction_adaptive_histogram(vehicle)
            vertexid = vehicle.vertexid
            vertex_info = {
                "vertexid": vertexid,
                "vehicle_id": vehicle.id,
                "hist": hist
            }
            response_data["leaving_vehicles"].append(vertex_info)
        
        # --- TIMING: Process entering vehicles ---
        entering_feature_start = time.time()
        metrics["timings"]["leaving_features"] = entering_feature_start - leaving_feature_start
        
        # Process entering vehicles
        for vehicle in entering_vehicles:
            logging.info("Vehicle ID: %d is entering", vehicle.id)
            hist = feature_extraction_adaptive_histogram(vehicle)
            vertexid = vehicle.vertexid
            vertex_info = {
                "vertexid": vertexid,
                "vehicle_id": vehicle.id,
                "hist": hist
            }
            response_data["entering_vehicles"].append(vertex_info)
        
        # --- TIMING: Storage ---
        storage_start = time.time()
        metrics["timings"]["entering_features"] = storage_start - entering_feature_start
        
        # Store data to in-memory storage
        start_upload_time = time.time()
        in_memory_storage[camera_id] = vt
        end_upload_time = time.time()
        upload_time = end_upload_time - start_upload_time
        
        # Measure size of stored data
        stored_size_kb = get_size_kb(vt)
        logging.info("Stored object size: %.2f KB", stored_size_kb)
        logging.info("In-memory Storage Time: %.6f seconds", upload_time)

        # --- TIMING: Cleanup ---
        cleanup_start = time.time()
        metrics["timings"]["storage"] = cleanup_start - storage_start
        
        # Cleanup
        tgraph.shutdown()
        logging.info("Finished Iteration")

        # --- TIMING: Response preparation ---
        response_start = time.time()
        metrics["timings"]["cleanup"] = response_start - cleanup_start
        
        # Calculate total processing time
        end_time = time.time()
        latency = end_time - start_time
        logging.info("End-to-end latency: %.6f seconds", latency)

        # Add performance metrics to response
        response_data["download_time"] = download_time
        response_data["upload_time"] = upload_time
        response_data["retrieved_size_kb"] = retrieved_size_kb
        response_data["stored_size_kb"] = stored_size_kb
        response_data["processing_time"] = latency
        
        # Add detailed metrics
        metrics["timings"]["response_prep"] = time.time() - response_start
        metrics["timings"]["total"] = latency
        metrics["sizes"]["retrieved_kb"] = retrieved_size_kb
        metrics["sizes"]["stored_kb"] = stored_size_kb
        
        response_data["detailed_metrics"] = metrics
        
        # Return the final response
        response = {
            "statusCode": 200,
            "body": jsonpickle.encode(response_data)
        }
        return response

    except Exception as e:
        # Calculate how far we got in the process
        end_time = time.time()
        metrics["timings"]["error_point"] = end_time - start_time
        
        logging.error("Unhandled exception: %s", str(e))
        logging.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "partial_metrics": metrics
            })
        }
    
if __name__ == "__main__":
    # Create a test image if the file doesn't exist
    import os
    import numpy as np
    
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Create a mock test image if it doesn't exist
    if not os.path.exists('coldstart.jpeg'):
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        test_img[100:200, 100:200] = 255
        cv2.imwrite('coldstart.jpeg', test_img)
        print("Created test image 'coldstart.jpeg'")
    
    # Read the image properly
    try:
        with open('coldstart.jpeg', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            print("Successfully loaded test image")
    except Exception as e:
        print(f"Error loading image: {e}")
        blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpeg', blank_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        print("Created fallback blank image")
    
    # Create a mock event object
    class MockEvent:
        def __init__(self, body_data):
            self.body = body_data.encode('utf-8')
    
    # Test with a simple payload
    payload = json.dumps({
        "camera_id": "0",
        "frame_id": "1",
        "image": encoded_image,
        "bboxes": [[183.3546, 233.2469, 267.5851, 296.2590, 100]]
    })
    
    mock_event = MockEvent(payload)
    
    print("\n==== Testing Handler with OpenFaaS Format ====")
    result = handle(mock_event, None)
    print("==== Result ====")
    print(result)
    print("====================\n")
    
    # Also test with STateFlow format
    print("\n==== Testing with STateFlow Format ====")
    stateflow_payload = {
        "Payload": base64.b64encode(payload.encode('utf-8')).decode('utf-8')
    }
    result = handle(stateflow_payload, None)
    print("==== Result ====")
    print(result)
    print("====================\n")