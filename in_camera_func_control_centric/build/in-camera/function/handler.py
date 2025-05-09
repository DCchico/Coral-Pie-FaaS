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
import redis

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

# Redis connection
redis_host = os.environ.get("REDIS_ADDR", "128.110.217.193")
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
try:
    redis_client.ping()
    logging.info(f"Connected to Redis at {redis_host}:{redis_port}")
except redis.exceptions.ConnectionError as e:
    logging.error(f"Redis connection error: {e}")
    raise  # Fail if Redis is not available - no fallback

# Function to get size of object in KB
def get_size_kb(obj):
    try:
        # Use pickle to get a more accurate size of complex objects
        return len(pickle.dumps(obj)) / 1024.0
    except:
        # Fallback to sys.getsizeof if pickle fails
        return sys.getsizeof(obj) / 1024.0

# Enhanced Redis helper functions for ordered processing
def store_frame_state(camera_id, frame_id, obj):
    """Store state for a specific camera and frame"""
    try:
        serialized_obj = pickle.dumps(obj)
        
        # Store using camera_id:frame_id as key
        key = f"incamera:{camera_id}:{frame_id}"
        
        # Store the state
        result = redis_client.set(key, serialized_obj)
        
        # Additionally, update a pointer to the latest frame
        redis_client.set(f"incamera:{camera_id}:latest", str(frame_id))
        
        return result, len(serialized_obj)
    except Exception as e:
        logging.error(f"Redis storage error: {e}")
        raise

def get_previous_frame_state(camera_id, current_frame_id, max_wait_seconds=10):
    """
    Get state from the previous frame, waiting if necessary.
    
    Args:
        camera_id: Camera identifier
        current_frame_id: Current frame ID
        max_wait_seconds: Maximum time to wait for previous frame, default 10 seconds
        
    Returns:
        (retrieved_object, size_in_bytes)
    """
    try:
        retrieval_metrics = {
            "initial_check_time": 0,
            "pubsub_setup_time": 0,
            "waiting_time": 0,
            "final_retrieval_time": 0,
            "already_available": 0,
            "wait_cycles": 0,
            "key_found_after_wait": 0,
            "timed_out": 0
        }
        
        # Calculate the previous frame ID
        prev_frame_id = int(current_frame_id) - 1
        
        # For frame 0 or frame 1, there's no previous state to retrieve
        if prev_frame_id <= 0:
            logging.info(f"No previous state expected for first frame of camera {camera_id}")
            return None, 0, retrieval_metrics
            
        # Key for the previous frame's state
        key = f"incamera:{camera_id}:{prev_frame_id}"
        
        # First, check if it's already available
        initial_check_start = time.time()
        serialized_obj = redis_client.get(key)
        retrieval_metrics["initial_check_time"] = (time.time() - initial_check_start) * 1000  # ms
        logging.debug(f"Initial check time: {retrieval_metrics['initial_check_time']:.2f}ms")
        
        if serialized_obj:
            logging.info(f"Previous frame {prev_frame_id} was immediately available")
            obj = pickle.loads(serialized_obj)
            size = len(serialized_obj)
            retrieval_metrics["already_available"] = 1
            return obj, size, retrieval_metrics
            
        # If not available, use Redis keyspace notifications to wait for it
        # Create a Redis pubsub object
        pubsub_setup_start = time.time()
        pubsub = redis_client.pubsub()
        
        # Subscribe to keyspace notifications for this key
        notification_key = f"__keyspace@0__:{key}"
        pubsub.subscribe(notification_key)
        
        # Enable keyspace notifications if not already enabled
        redis_client.config_set('notify-keyspace-events', 'KA')
        retrieval_metrics["pubsub_setup_time"] = (time.time() - pubsub_setup_start) * 1000  # ms
        logging.debug(f"PubSub setup time: {retrieval_metrics['pubsub_setup_time']:.2f}ms")
            
        logging.info(f"Waiting for previous frame {prev_frame_id} for camera {camera_id} (max {max_wait_seconds}s)")
        
        # Wait for the key to be created (with timeout)
        waiting_start = time.time()
        wait_start = time.time()
        serial_obj = None
        while time.time() - wait_start < max_wait_seconds:
            retrieval_metrics["wait_cycles"] += 1
            message = pubsub.get_message(timeout=0.1)
            if message and message['type'] == 'message' and message['data'] == b'set':
                # Key was created, retrieve it
                retrieval_metrics["waiting_time"] = (time.time() - waiting_start) * 1000  # ms
                final_retrieval_start = time.time()
                serialized_obj = redis_client.get(key)
                retrieval_metrics["final_retrieval_time"] = (time.time() - final_retrieval_start) * 1000  # ms
            else:
                final_retrieval_start = time.time()
                serialized_obj = redis_client.get(key)
                if serialized_obj:
                    retrieval_metrics["final_retrieval_time"] = (time.time() - final_retrieval_start) * 1000  # ms
                    retrieval_metrics["waiting_time"] = (time.time() - waiting_start - retrieval_metrics["final_retrieval_time"]/1000) * 1000  # ms
                
            if serialized_obj:
                obj = pickle.loads(serialized_obj)
                size = len(serialized_obj)
                wait_time = time.time() - wait_start
                logging.info(f"Previous frame {prev_frame_id} became available after {wait_time:.2f}s")
                retrieval_metrics["key_found_after_wait"] = 1
                pubsub.unsubscribe()
                
                logging.debug(f"Full retrieval metrics: {retrieval_metrics}")
                return obj, size, retrieval_metrics
            
        
        # If we get here, we timed out waiting
        pubsub.unsubscribe()
        retrieval_metrics["waiting_time"] = (time.time() - waiting_start) * 1000  # ms
        retrieval_metrics["timed_out"] = 1
        logging.warning(f"Timed out waiting for previous frame {prev_frame_id} for camera {camera_id}")
        
        logging.debug(f"Timed out retrieval metrics: {retrieval_metrics}")
        return None, 0, retrieval_metrics
        
    except Exception as e:
        logging.error(f"Redis retrieval error: {e}")
        raise

'''
def clean_old_states(camera_id, current_frame_id, keep_frames=5):
    """Clean up old states, keeping only the most recent few frames"""
    try:
        # Only delete frames older than (current - keep_frames)
        oldest_to_keep = int(current_frame_id) - keep_frames
        
        if oldest_to_keep <= 0:
            return  # Not enough frames processed yet to clean up
            
        # Delete all frames older than oldest_to_keep
        for frame_to_delete in range(1, oldest_to_keep):
            key = f"incamera:{camera_id}:{frame_to_delete}"
            deleted = redis_client.delete(key)
            if deleted:
                logging.debug(f"Cleaned up old state: {key}")
    except Exception as e:
        logging.error(f"Redis cleanup error: {e}")
        # Don't fail the main processing if cleanup fails
        pass
'''

def handle(req, context):
    global tgraph, vt
    """
    Handler function that processes vehicle tracking from images
    """
    start_time = time.time()  # Record the overall start time
    metrics = {
        "timings": {},
        "sizes": {},
        "retrieval_details": {},
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
            logging.info(f"Processing frame {frame_id} for camera {camera_id}")
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
        if not bboxes:
            logging.error("Missing bboxes field")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing bboxes field"})
            }

        # Initialize response data
        response_data = {
            "camera_id": camera_id,
            "frame_id": frame_id,
            "leaving_vehicles": [],
            "entering_vehicles": []
        }

        # Check for test_mode which skips graph operations
        if input_data.get('test_mode', False):
            logging.info("TEST MODE ACTIVATED - skipping graph operations")
            # Return a simple response without any graph processing
            response_data = {
                "message": "Test mode successful",
                "camera_id": camera_id,
                "frame_id": frame_id,
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
        
        # --- TIMING: State retrieval ---
        retrieve_start = time.time()
        metrics["timings"]["initialization"] = retrieve_start - init_start
        
        # Retrieve the VehicleTracking object from the previous frame
        start_download_time = time.time()
        retrieved_vt, size_bytes, retrieval_details = get_previous_frame_state(camera_id, frame_id)
        retrieved_size_kb = size_bytes / 1024.0
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        
        # Debug output to verify metrics are populated
        logging.info(f"Retrieval details: {retrieval_details}")
        
        # Store detailed retrieval metrics in proper nested structure
        metrics["retrieval_details"] = {
            "retrieval": retrieval_details
        }
        
        # Log more specific information about the retrieval process
        if retrieval_details:
            if retrieval_details.get("already_available", False):
                logging.info(f"Frame was immediately available. Initial check: {retrieval_details.get('initial_check_time', 0):.2f}ms")
            elif retrieval_details.get("key_found_after_wait", False):
                logging.info(f"Frame was found after waiting. Setup: {retrieval_details.get('pubsub_setup_time', 0):.2f}ms, " +
                             f"Wait: {retrieval_details.get('waiting_time', 0):.2f}ms, " +
                             f"Final retrieval: {retrieval_details.get('final_retrieval_time', 0):.2f}ms, " +
                             f"Wait cycles: {retrieval_details.get('wait_cycles', 0)}")
            elif retrieval_details.get("timed_out", False):
                logging.info(f"Timed out waiting for frame. Setup: {retrieval_details.get('pubsub_setup_time', 0):.2f}ms, " +
                             f"Wait time: {retrieval_details.get('waiting_time', 0):.2f}ms")

        logging.info(f"State retrieval time: {download_time:.6f} seconds")

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
            logging.info(f"Successfully decoded image, shape: {image.shape}")
        except Exception as e:
            logging.error(f"Error decoding image: {e}")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Image decoding error: {str(e)}"})
            }

        # --- TIMING: Sort update (tracking) ---
        sort_start = time.time()
        metrics["timings"]["image_decoding"] = sort_start - decode_start
        metrics["sizes"]["image_kb"] = len(decoded_image) / 1024.0
        
        # Use retrieved or create new if not found
        current_vt = retrieved_vt if retrieved_vt else VehicleTracking()
        
        # Process bboxes for tracking
        trackers = current_vt.sort_update(np.asarray(bboxes))
        
        # --- TIMING: Status update ---
        status_start = time.time()
        metrics["timings"]["sort_update"] = status_start - sort_start
        
        # Update status with camera ID, frame ID, and frame
        status = current_vt.status_update(tgraph, camera_id, frame_id, image)
        
        # --- TIMING: Process leaving vehicles ---
        leaving_feature_start = time.time()
        metrics["timings"]["status_update"] = leaving_feature_start - status_start
        
        entering_vehicles = status["entering_vehicles"]
        leaving_vehicle = status["leaving_vehicle"]
        
        # Process leaving vehicles
        for vehicle in leaving_vehicle:
            logging.info(f"Vehicle ID: {vehicle.id} is leaving")
            # Assuming feature_extraction_adaptive_histogram is defined elsewhere
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
            logging.info(f"Vehicle ID: {vehicle.id} is entering")
            hist = feature_extraction_adaptive_histogram(vehicle)
            vertexid = vehicle.vertexid
            vertex_info = {
                "vertexid": vertexid,
                "vehicle_id": vehicle.id,
                "hist": hist
            }
            response_data["entering_vehicles"].append(vertex_info)
        
        # --- TIMING: State storage ---
        storage_start = time.time()
        metrics["timings"]["entering_features"] = storage_start - entering_feature_start
        
        # Store updated state for this frame
        start_upload_time = time.time()
        
        # Store the current frame's state
        success, size_bytes = store_frame_state(camera_id, frame_id, current_vt)
        stored_size_kb = size_bytes / 1024.0
            
        end_upload_time = time.time()
        upload_time = end_upload_time - start_upload_time
        
        logging.info(f"Stored state for camera {camera_id}, frame {frame_id}, size: {stored_size_kb:.2f} KB")
        logging.info(f"State storage time: {upload_time:.6f} seconds")
        
        # Clean up old states (keep last 5 frames)
        # clean_old_states(camera_id, frame_id, keep_frames=5)

        # --- TIMING: Cleanup ---
        cleanup_start = time.time()
        metrics["timings"]["storage"] = cleanup_start - storage_start
        
        # No need to shut down tgraph for every request
        logging.info("Processing complete")

        # --- TIMING: Response preparation ---
        response_start = time.time()
        metrics["timings"]["cleanup"] = response_start - cleanup_start
        
        # Calculate total processing time
        end_time = time.time()
        latency = end_time - start_time
        logging.info(f"End-to-end latency: {latency:.6f} seconds")

        # Add performance metrics to response
        response_data["download_time"] = download_time
        response_data["upload_time"] = upload_time
        response_data["retrieved_size_kb"] = retrieved_size_kb
        response_data["stored_size_kb"] = stored_size_kb
        response_data["processing_time"] = latency
        response_data["storage_type"] = "redis"
        
        # Add detailed metrics correctly structured
        metrics["timings"]["response_prep"] = time.time() - response_start
        metrics["timings"]["total"] = latency
        metrics["sizes"]["retrieved_kb"] = retrieved_size_kb
        metrics["sizes"]["stored_kb"] = stored_size_kb
        
        response_data["detailed_metrics"] = metrics
        
        # Also include it at top level for flexibility
        response_data["retrieval_metrics"] = retrieval_details
        
        # Return the final response
        response = {
            "statusCode": 200,
            "body": jsonpickle.encode(response_data)
        }
        return response

    except redis.exceptions.RedisError as e:
        # Redis error handling
        logging.error(f"Redis error: {str(e)}")
        logging.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Redis error: {str(e)}",
                "traceback": traceback.format_exc()
            })
        }
    except (ConnectionError, KeyError) as e:
        # Handle JanusGraph/Gremlin connection issues
        logging.error(f"Graph database error: {str(e)}")
        logging.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Graph database error: {str(e)}",
                "traceback": traceback.format_exc()
            })
        }
    except Exception as e:
        # General error handling
        end_time = time.time()
        metrics["timings"]["error_point"] = end_time - start_time
        
        logging.error(f"Unhandled exception: {str(e)}")
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