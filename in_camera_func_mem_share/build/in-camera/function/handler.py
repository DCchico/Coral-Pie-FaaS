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
from multiprocessing import shared_memory
import struct

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

# Replace Redis imports
# import redis
# redis_host = os.environ.get("REDIS_ADDR", "128.110.217.193")
# # redis_host = "128.110.217.193"
# redis_port = 6379
# redis_password = None  # Add password if your Redis requires authentication

# Replace Redis initialization
# redis_client = redis.Redis(
#     host=redis_host,
#     port=redis_port,
#     password=redis_password,
#     socket_timeout=5,
#     decode_responses=False  # We need binary data for pickled objects
# )
# try:
#     redis_client.ping()
#     logging.info(f"Connected to Redis at {redis_host}:{redis_port}")
# except redis.exceptions.ConnectionError as e:
#     logging.error(f"Redis connection error: {e}")
#     raise  # Fail if Redis is not available - no fallback

# Function to get size of object in KB
def get_size_kb(obj):
    try:
        # Use pickle to get a more accurate size of complex objects
        return len(pickle.dumps(obj)) / 1024.0
    except:
        # Fallback to sys.getsizeof if pickle fails
        return sys.getsizeof(obj) / 1024.0

# Replace Redis initialization
# redis_client = redis.Redis(...)
camera_states = {}  # Dictionary to track shared memory states by camera

def store_frame_state(camera_id, frame_id, tracking_obj):
    """Store state for a specific camera and frame using shared memory"""
    try:
        store_metrics = {
            "pickle_time": 0,
            "memory_write_time": 0,
            "header_write_time": 0,
            "obj_size_before": 0,
            "serialized_size": 0
        }
        
        # Get or create shared memory
        if camera_id not in camera_states:
            try:
                shm = shared_memory.SharedMemory(
                    name=f"vt_camera_{camera_id}", 
                    create=True, 
                    size=50*1024*1024
                )
                # Initialize header: [frame_id(8), state_size(8), valid_flag(8)]
                header = struct.pack("!QQQ", 0, 0, 0)
                shm.buf[:len(header)] = header
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=f"vt_camera_{camera_id}")
            camera_states[camera_id] = shm
        else:
            shm = camera_states[camera_id]

        # Measure object size before serialization
        store_metrics["obj_size_before"] = sys.getsizeof(tracking_obj)

        # Time pickle serialization
        pickle_start = time.time()
        serialized = pickle.dumps(tracking_obj)
        store_metrics["pickle_time"] = (time.time() - pickle_start) * 1000  # Convert to ms
        
        state_size = len(serialized)
        store_metrics["serialized_size"] = state_size
        
        if state_size > len(shm.buf) - 24:  # Check if it fits
            return False, 0, store_metrics

        # Time memory write operations separately
        write_start = time.time()
        shm.buf[24:24+state_size] = serialized
        store_metrics["memory_write_time"] = (time.time() - write_start) * 1000  # Convert to ms
        
        # Time header write
        header_start = time.time()
        shm.buf[:24] = struct.pack("!QQQ", frame_id, state_size, 1)
        store_metrics["header_write_time"] = (time.time() - header_start) * 1000  # Convert to ms
        
        logging.info(f"Store metrics - Pickle: {store_metrics['pickle_time']:.2f}ms, "
                    f"Memory Write: {store_metrics['memory_write_time']:.2f}ms, "
                    f"Header Write: {store_metrics['header_write_time']:.2f}ms, "
                    f"Object Size: {store_metrics['obj_size_before']} bytes, "
                    f"Serialized Size: {store_metrics['serialized_size']} bytes")
        
        return True, state_size, store_metrics
                
    except Exception as e:
        logging.error(f"Error storing frame state: {e}")
        return False, 0, {}

def get_previous_frame_state(camera_id, current_frame_id, max_wait_seconds=10):
    """Get state from the previous frame using shared memory"""
    try:
        retrieval_metrics = {
            "direct_access": 1,
            "wait_required": 0,
            "already_available": 0,
            "timed_out": 0
        }
        
        prev_frame_id = int(current_frame_id) - 1
        if prev_frame_id <= 0:
            return None, 0, retrieval_metrics
            
        # Get or connect to shared memory
        if camera_id not in camera_states:
            try:
                shm = shared_memory.SharedMemory(name=f"vt_camera_{camera_id}")
                camera_states[camera_id] = shm
            except FileNotFoundError:
                return None, 0, retrieval_metrics
        else:
            shm = camera_states[camera_id]
            
        # Read header
        header = bytes(shm.buf[:24])
        frame_id, state_size, valid = struct.unpack("!QQQ", header)
        
        if valid == 1 and frame_id == prev_frame_id:
            # Read and deserialize state
            serialized = bytes(shm.buf[24:24+state_size])
            state = pickle.loads(serialized)
            retrieval_metrics["already_available"] = 1
            return state, state_size, retrieval_metrics
            
        retrieval_metrics["timed_out"] = 0
        return None, 0, retrieval_metrics
        
    except Exception as e:
        logging.error(f"Error retrieving frame state: {e}")
        return None, 0, retrieval_metrics

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
    # Start total time measurement
    total_start_time = time.time()
    pre_state_start_time = time.time()
    
    # Initialize monitoring metrics (will calculate at the end)
    monitoring_metrics = {
        "total_time": 0,
        "pre_state_processing": 0,
        "post_state_processing": 0,
        "state_handling": {
            "level": "L2",  # Since we're using shared memory
            "process_time": 0,
            "idle_time": 0
        }
    }

    # Keep existing metrics structure
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
                    "total": time.time() - total_start_time
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
        
        # --- TIMING: Image decoding (moved up) ---
        decode_start = time.time()
        metrics["timings"]["initialization"] = decode_start - init_start
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
        
        metrics["timings"]["image_decoding"] = time.time() - decode_start
        metrics["sizes"]["image_kb"] = len(decoded_image) / 1024.0

        # Mark end of pre-state processing and start of state handling
        pre_state_end_time = time.time()
        state_handling_start = time.time()

        # --- TIMING: State retrieval ---
        retrieve_start = time.time()
        
        # Retrieve the VehicleTracking object from the previous frame
        start_download_time = time.time()
        retrieved_vt, size_bytes, retrieval_details = get_previous_frame_state(camera_id, frame_id)
        retrieved_size_kb = size_bytes / 1024.0
        end_download_time = time.time()
        download_time = end_download_time - start_download_time
        
        metrics["timings"]["retrieval"] = time.time() - retrieve_start
        
        # Calculate idle time from retrieval metrics
        idle_time = 0
        if retrieval_details.get("already_available") == 0:
            idle_time = retrieval_details.get("waiting_time", 0)

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

        # --- TIMING: Sort update (tracking) ---
        sort_start = time.time()
        
        # Use retrieved or create new if not found
        current_vt = retrieved_vt if retrieved_vt else VehicleTracking()
        
        # Process bboxes for tracking
        trackers = current_vt.sort_update(np.asarray(bboxes))
        
        # --- TIMING: Status update ---
        status_start = time.time()
        metrics["timings"]["sort_update"] = status_start - sort_start
        
        # Update status with camera ID, frame ID, and frame
        status = current_vt.status_update(tgraph, camera_id, frame_id, image)
        
        # --- TIMING: State storage (moved up) ---
        storage_start = time.time()
        metrics["timings"]["status_update"] = storage_start - status_start
        
        # Store the current frame's state
        store_start_time = time.time()
        success, size_bytes, store_metrics = store_frame_state(camera_id, frame_id, current_vt)
        store_end_time = time.time()
        stored_size_kb = size_bytes / 1024.0
            
        # Mark end of state handling and start of post-processing
        state_handling_end = time.time()
        post_state_start = time.time()
        
        leaving_feature_start = time.time()
        metrics["timings"]["storage"] = post_state_start - storage_start
        
        # --- TIMING: Process leaving/entering vehicles (moved after storage) ---
        entering_vehicles = status["entering_vehicles"]
        leaving_vehicle = status["leaving_vehicle"]
        
        # Process leaving vehicles
        for vehicle in leaving_vehicle:
            logging.info(f"Vehicle ID: {vehicle.id} is leaving")
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
        metrics["timings"]["entering_features"] =  time.time() - entering_feature_start
        
        logging.info(f"Stored state for camera {camera_id}, frame {frame_id}, size: {stored_size_kb:.2f} KB")
        logging.info(f"State storage time: {(store_end_time - store_start_time):.6f} seconds")
        
        # Add store metrics to the response
        metrics["store_details"] = store_metrics

        # Clean up old states (keep last 5 frames)
        # clean_old_states(camera_id, frame_id, keep_frames=5)

        # --- TIMING: Cleanup ---
        cleanup_start = time.time()
        metrics["timings"]["cleanup"] = 0
        
        # No need to shut down tgraph for every request
        logging.info("Processing complete")

        # --- TIMING: Response preparation ---
        response_start = time.time()
        
        # Calculate total processing time
        end_time = time.time()
        latency = end_time - total_start_time
        logging.info(f"End-to-end latency: {latency:.6f} seconds")

        # Add performance metrics to response
        response_data["download_time"] = download_time
        response_data["upload_time"] = store_end_time - store_start_time
        response_data["retrieved_size_kb"] = retrieved_size_kb
        response_data["stored_size_kb"] = stored_size_kb
        response_data["processing_time"] = latency
        response_data["storage_type"] = "shared_memory"
        
        # Add detailed metrics correctly structured
        metrics["timings"]["response_prep"] = time.time() - response_start
        metrics["timings"]["total"] = latency
        metrics["sizes"]["retrieved_kb"] = retrieved_size_kb
        metrics["sizes"]["stored_kb"] = stored_size_kb
        
        response_data["detailed_metrics"] = metrics
        
        # Also include it at top level for flexibility
        response_data["retrieval_metrics"] = retrieval_details
                # Calculate pre_state_processing (everything before state retrieval)
        pre_state_time = (pre_state_end_time - pre_state_start_time)
        
        # Calculate post_state_processing (everything after state storage)
        post_state_time = (end_time - post_state_start)
        
        # Calculate state handling time (retrieval through storage)
        state_handling_time = (state_handling_end - state_handling_start)
        
        # Update monitoring metrics
        monitoring_metrics.update({
            "total_time": end_time - total_start_time,
            "pre_state_processing": pre_state_time,
            "post_state_processing": post_state_time,
            "state_handling": {
                "level": "L2",  # shared memory
                "process_time": state_handling_time,
                "idle_time": idle_time
            }
        })

        # Determine storage level based on storage type
        storage_type = response_data.get("storage_type", "unknown")
        if storage_type == "direct":
            monitoring_metrics["state_handling"]["level"] = "L1"
        elif storage_type == "shared_memory":
            monitoring_metrics["state_handling"]["level"] = "L2"
        elif storage_type == "redis":
            monitoring_metrics["state_handling"]["level"] = "L3"

        # Metrics are simple types, only need one conversion
        metrics_json = json.dumps(monitoring_metrics)
        
        response = {
            "statusCode": 200,
            "body": jsonpickle.encode(response_data),  # Need jsonpickle for complex objects
            "headers": {
                "X-Function-Metrics": metrics_json,  # Simple JSON is safe for headers
                "Content-Type": "application/json"
            }
        }
        return response

    except Exception as e:
        # General error handling
        end_time = time.time()
        metrics["timings"]["error_point"] = end_time - total_start_time
        
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
    
    finally:
        # Add this to the end of handle() function, just before returning the response
        def cleanup_shared_memory():
            """Clean up shared memory when service is shutting down"""
            for camera_id, shm in camera_states.items():
                try:
                    shm.close()
                    shm.unlink()  # This will remove the shared memory segment
                except Exception as e:
                    logging.error(f"Error cleaning up shared memory: {e}")

        # You might want to register this with atexit
        import atexit
        atexit.register(cleanup_shared_memory)

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