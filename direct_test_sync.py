"""
Test Redis frame synchronization by directly calling the handler function
"""
import base64
import json
import time
import threading
import random
import os
import sys
import cv2
import numpy as np

# Import the handler function
sys.path.append("./in_camera_func_control_centric/in-camera")
import handler

# Configuration
TEST_IMAGE_PATH = "camera_stream_data/000000.jpeg"  # Path to your test image
CAMERA_ID = "test-camera"     # Use a specific test camera ID

def load_test_image(image_path):
    """Load and encode a test image as base64"""
    if not os.path.exists(image_path):
        print(f"ERROR: Test image not found at {image_path}")
        # Create a small test image instead
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img = cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("generated_test.jpg", img)
        image_path = "generated_test.jpg"
        print(f"Created a test image at {image_path}")
    
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        return base64.b64encode(img_bytes).decode('utf-8')

def create_mock_event(camera_id, frame_id, image_base64):
    """Create a mock event object to pass to the handler"""
    body = json.dumps({
        "camera_id": camera_id,
        "frame_id": frame_id,
        "image": image_base64,
        "bboxes": [[100, 100, 200, 200, 0.95]],  # Simple test bbox
        "test_mode": True  # Enable test mode to skip graph operations
    })
    
    class MockEvent:
        def __init__(self, body_data):
            self.body = body_data
    
    return MockEvent(body)

def send_frame(camera_id, frame_id, image_base64):
    """Send a frame by directly calling the handler function"""
    event = create_mock_event(camera_id, frame_id, image_base64)
    
    start_time = time.time()
    try:
        print(f"Processing frame {frame_id}...")
        result = handler.handle(event, None)
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"Frame {frame_id} response time: {response_time:.3f}s")
        
        if result.get('statusCode') == 200:
            try:
                # Parse the response body
                response_data = json.loads(result.get('body', '{}'))
                processing_time = response_data.get('processing_time', 0)
                waiting_time = response_time - processing_time
                
                return {
                    "frame_id": frame_id,
                    "response_time": response_time,
                    "processing_time": processing_time,
                    "waiting_time": waiting_time,
                    "success": True
                }
            except json.JSONDecodeError:
                print(f"ERROR: Invalid JSON response for frame {frame_id}")
        else:
            print(f"ERROR: Non-200 status code for frame {frame_id}: {result.get('statusCode')}")
            print(f"Response: {result.get('body')}")
        
        return {
            "frame_id": frame_id,
            "response_time": response_time,
            "status_code": result.get('statusCode', 500),
            "error": result.get('body', 'Unknown error'),
            "success": False
        }
    except Exception as e:
        end_time = time.time()
        print(f"ERROR processing frame {frame_id}: {e}")
        return {
            "frame_id": frame_id,
            "response_time": end_time - start_time,
            "error": str(e),
            "success": False
        }

def test_sequential_frames(camera_id, image_base64, num_frames=5):
    """Test sending frames in sequential order (1, 2, 3, 4, 5)"""
    print("\n===== TESTING SEQUENTIAL FRAME PROCESSING =====")
    results = []
    
    for frame_id in range(1, num_frames+1):
        result = send_frame(camera_id, frame_id, image_base64)
        results.append(result)
        time.sleep(0.5)  # Short delay between frames
    
    print_results(results, "Sequential Frames")
    return results

def test_reverse_frames(camera_id, image_base64, num_frames=5):
    """Test sending frames in reverse order (5, 4, 3, 2, 1)"""
    print("\n===== TESTING REVERSE FRAME PROCESSING =====")
    results = []
    
    for frame_id in range(num_frames, 0, -1):
        result = send_frame(camera_id, frame_id, image_base64)
        results.append(result)
        time.sleep(0.5)  # Short delay between frames
    
    # Sort results by frame_id for easier reading
    results.sort(key=lambda r: r['frame_id'])
    print_results(results, "Reverse Frames")
    return results

def print_results(results, test_name):
    """Print formatted test results"""
    print(f"\n----- {test_name} Results -----")
    print(f"{'Frame ID':<10} {'Success':<10} {'Response Time':<15} {'Processing Time':<15} {'Waiting Time':<15}")
    print("-" * 70)
    
    for result in results:
        if result:
            frame_id = result.get('frame_id', 'N/A')
            success = result.get('success', False)
            response_time = f"{result.get('response_time', 0):.3f}s"
            processing_time = f"{result.get('processing_time', 0):.3f}s"
            waiting_time = f"{result.get('waiting_time', 0):.3f}s"
            
            print(f"{frame_id:<10} {str(success):<10} {response_time:<15} {processing_time:<15} {waiting_time:<15}")
    
    # Calculate averages for successful requests
    successful = [r for r in results if r and r.get('success', False)]
    if successful:
        avg_response = sum(r.get('response_time', 0) for r in successful) / len(successful)
        avg_processing = sum(r.get('processing_time', 0) for r in successful) / len(successful)
        avg_waiting = sum(r.get('waiting_time', 0) for r in successful) / len(successful)
        
        print("-" * 70)
        print(f"{'AVERAGE':<10} {'':<10} {f'{avg_response:.3f}s':<15} {f'{avg_processing:.3f}s':<15} {f'{avg_waiting:.3f}s':<15}")
    
    print()

def main():
    """Run all tests"""
    # Load test image
    image_path = TEST_IMAGE_PATH
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    print(f"Loading test image from {image_path}...")
    image_base64 = load_test_image(image_path)
    
    # Unique camera ID for each test run to avoid interference
    timestamp = int(time.time())
    unique_camera_id = f"{CAMERA_ID}-{timestamp}"
    
    # Run tests
    try:
        test_sequential_frames(unique_camera_id, image_base64, num_frames=3)
        time.sleep(2)  # Wait between tests
        
        test_reverse_frames(f"{unique_camera_id}-reverse", image_base64, num_frames=3)
        
    except KeyboardInterrupt:
        print("Tests interrupted by user.")
    except Exception as e:
        print(f"Error running tests: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 