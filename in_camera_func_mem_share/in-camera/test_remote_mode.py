#!/usr/bin/env python3
"""
Test script for remote mode with StateProxy
Demonstrates how to use the handler with remote storage mode.
"""

import json
import base64
import numpy as np
import cv2
import time
import os

def create_test_request(storage_mode="remote", remote_host="localhost", remote_port=8080):
    """Create a test request for the handler"""
    
    # Create a simple test image
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    test_img[100:200, 100:200] = 255  # White rectangle
    
    # Encode the image
    _, buffer = cv2.imencode('.jpeg', test_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    # Create test bboxes
    bboxes = [[183.3546, 233.2469, 267.5851, 296.2590, 100]]
    
    # Create the request payload
    payload = {
        "camera_id": "test_camera_0",
        "frame_id": 123,
        "image": encoded_image,
        "bboxes": bboxes,
        "storage_mode": storage_mode,  # 'direct', 'shared_memory', 'remote'
        "test_mode": False
    }
    
    return payload

def create_mock_event(payload, storage_mode="shared_memory", dependent_state_mode=""):
    """Create a mock event with headers"""
    
    class MockEvent:
        def __init__(self, body_data, storage_mode, dependent_state_mode):
            self.body = json.dumps(body_data).encode('utf-8')
            self.headers = {
                'state-location': storage_mode,
                'dependent-state-location': dependent_state_mode
            }
    
    return MockEvent(payload, storage_mode, dependent_state_mode)

def test_local_mode():
    """Test local shared memory mode"""
    print("=== Testing Local Shared Memory Mode ===")
    
    payload = create_test_request(storage_mode="shared_memory")
    mock_event = create_mock_event(payload, storage_mode="shared_memory")
    
    # Import and call the handler
    from handler import handle
    
    start_time = time.time()
    result = handle(mock_event, None)
    end_time = time.time()
    
    print(f"Local mode execution time: {end_time - start_time:.3f}s")
    print(f"Status code: {result.get('statusCode', 'N/A')}")
    
    if result.get('statusCode') == 200:
        print("✅ Local mode test successful")
    else:
        print("❌ Local mode test failed")
        print(f"Error: {result.get('body', 'Unknown error')}")

def test_remote_mode(remote_host="localhost", remote_port=8080):
    """Test remote mode with StateProxy"""
    print(f"=== Testing Remote Mode with StateProxy at {remote_host}:{remote_port} ===")
    
    payload = create_test_request(storage_mode="remote", remote_host=remote_host, remote_port=remote_port)
    
    # Create remote dependent state mode header
    dependent_state_mode = f"remote:{remote_host}:{remote_port}"
    mock_event = create_mock_event(payload, storage_mode="remote", dependent_state_mode=dependent_state_mode)
    
    # Import and call the handler
    from handler import handle
    
    start_time = time.time()
    result = handle(mock_event, None)
    end_time = time.time()
    
    print(f"Remote mode execution time: {end_time - start_time:.3f}s")
    print(f"Status code: {result.get('statusCode', 'N/A')}")
    
    if result.get('statusCode') == 200:
        print("✅ Remote mode test successful")
        
        # Parse the response to show metrics
        try:
            response_body = json.loads(result.get('body', '{}'))
            metrics = response_body.get('detailed_metrics', {})
            
            # Show storage information
            retrieval_metrics = metrics.get('retrieval_details', {}).get('retrieval', {})
            store_metrics = metrics.get('store_details', {})
            
            print(f"Retrieval remote operation: {retrieval_metrics.get('remote_operation', False)}")
            print(f"Store remote operation: {store_metrics.get('remote_operation', False)}")
            print(f"Retrieval remote time: {retrieval_metrics.get('remote_operation_time', 0):.2f}ms")
            print(f"Store remote time: {store_metrics.get('remote_operation_time', 0):.2f}ms")
            
        except Exception as e:
            print(f"Could not parse response metrics: {e}")
    else:
        print("❌ Remote mode test failed")
        print(f"Error: {result.get('body', 'Unknown error')}")

def test_direct_mode():
    """Test direct mode (no state persistence)"""
    print("=== Testing Direct Mode ===")
    
    payload = create_test_request(storage_mode="direct")
    mock_event = create_mock_event(payload, storage_mode="direct")
    
    # Import and call the handler
    from handler import handle
    
    start_time = time.time()
    result = handle(mock_event, None)
    end_time = time.time()
    
    print(f"Direct mode execution time: {end_time - start_time:.3f}s")
    print(f"Status code: {result.get('statusCode', 'N/A')}")
    
    if result.get('statusCode') == 200:
        print("✅ Direct mode test successful")
    else:
        print("❌ Direct mode test failed")
        print(f"Error: {result.get('body', 'Unknown error')}")

def main():
    """Main test function"""
    print("🚀 Testing Handler with Different Storage Modes")
    print("=" * 50)
    
    # Test direct mode first (should always work)
    test_direct_mode()
    print()
    
    # Test local shared memory mode
    test_local_mode()
    print()
    
    # Test remote mode (requires StateProxy to be running)
    remote_host = os.environ.get("STATE_PROXY_HOST", "localhost")
    remote_port = int(os.environ.get("STATE_PROXY_PORT", "8080"))
    test_remote_mode(remote_host, remote_port)
    print()
    
    print("🏁 All tests completed!")

if __name__ == "__main__":
    main() 