import time
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms.functional as transform
from PIL import Image
import io
import json
import base64

# Load the model and weights
start_model_loading = time.time()
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.3)
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
end_model_loading = time.time()

def handle(event, context):
    global start_model_loading, end_model_loading
    
    # Start total time measurement at the very beginning
    total_start_time = time.time()
    
    try:
        # Parse input JSON first
        if isinstance(event.body, bytes):
            body_str = event.body.decode('utf-8')
        else:
            body_str = str(event.body)
        
        input_data = json.loads(body_str)
        
        # Extract fields
        camera_id = input_data.get('camera_id')
        frame_id = int(input_data.get('frame_id'))
        encoded_image = input_data.get('image')
        
        # Decode base64 image
        image_data = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(image_data))
        img = transform.to_tensor(image)

        # Apply inference preprocessing and run detection
        preprocess = weights.transforms()
        start_preprocess = time.time()
        batch = [preprocess(img)]
        end_preprocess = time.time()

        # Run model prediction
        start_prediction = time.time()
        prediction = model(batch)[0]
        end_prediction = time.time()

        # Extract boxes, scores, and labels like in local_detector
        boxes = prediction["boxes"].detach().cpu().numpy().tolist()
        scores = prediction["scores"].detach().cpu().numpy().tolist()
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        
        # Filter by confidence threshold and format boxes
        bboxes = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > 0.3:  # Same threshold as local_detector
                # Format: [x1, y1, x2, y2, score] - matching local_detector's go_format_boxes
                bboxes.append(box + [score])
        
        # Format metrics
        detection_metrics = {
            "model_loading_time": (end_model_loading - start_model_loading) * 1000.0,
            "preprocessing_time": (end_preprocess - start_preprocess) * 1000.0,
            "prediction_time": (end_prediction - start_prediction) * 1000.0
        }
        end_model_loading = start_model_loading

        # Format output for in-camera function
        in_camera_input = {
            "camera_id": camera_id,
            "frame_id": frame_id,
            "image": encoded_image,
            "bboxes": bboxes
        }

        # Calculate total processing time at the end
        end_time = time.time()
        total_time = end_time - total_start_time

        # Create monitoring metrics structure
        monitoring_metrics = {
            "total_time": total_time,
            "pre_state_processing": 0,  # Always 0 for stateless function
            "post_state_processing": 0,  # Always 0 for stateless function
        }

        # Convert metrics to JSON string for header
        metrics_json = json.dumps(monitoring_metrics)

        # Return response with metrics in header
        return {
            "statusCode": 200,
            "body": json.dumps(in_camera_input),
            "headers": {
                "Content-Type": "application/json",
                "X-Function-Metrics": metrics_json  # Add metrics to header
            }
        }
    except Exception as e:
        # Add metrics even in error case
        end_time = time.time()
        error_metrics = {
            "total_time": end_time - total_start_time,
            "pre_state_processing": 0,
            "post_state_processing": 0,
        }
        
        error_response = {
            "error": f"Internal Server Error: {str(e)}"
        }
        return {
            "statusCode": 500,
            "body": json.dumps(error_response),
            "headers": {
                "Content-Type": "application/json",
                "X-Function-Metrics": json.dumps(error_metrics)
            }
        }

'''
if __name__ == "__main__":
    import os
    import glob
    
    # Define MockEvent class
    class MockEvent:
        def __init__(self, body_data):
            self.body = body_data.encode('utf-8')
    
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Using device: {device}")
    print(f"Model loaded in {(end_model_loading - start_model_loading):.2f} seconds")
    
    # Path to camera stream data
    IMAGE_DIR = "/usr/local/src/Coral-Pie-FaaS/camera_stream_data"
    
    # Get all jpeg files and sort them
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        exit(1)
    
    print(f"Found {len(image_files)} images")
    
    # Process each frame sequentially
    for image_path in image_files:
        # Extract frame_id from filename (000000.jpeg -> 0)
        frame_id = int(os.path.basename(image_path).split('.')[0])
        
        # Read and encode the image
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create test payload
        test_payload = {
            "camera_id": "camera1",
            "frame_id": frame_id,
            "image": encoded_image
        }
        
        # Create mock event and process frame
        mock_event = MockEvent(json.dumps(test_payload))
        result = handle(mock_event, None)
        
        # Print function response
        print(f"\n=== Frame {frame_id} Response ===")
        response_data = json.loads(result["body"].decode('utf-8'))
        print(f"Status Code: {result['statusCode']}")
        print(f"Camera ID: {response_data['camera_id']}")
        print(f"Frame ID: {response_data['frame_id']}")
        print(f"Number of detections: {len(response_data['bboxes'])}")
        
        if response_data['bboxes']:
            print("Detections:")
            for bbox in response_data['bboxes']:
                print(f"  {bbox}")  # Raw bbox values
        
        print("Detection Metrics:")
        for key, value in response_data['detection_metrics'].items():
            print(f"  {key}: {value:.2f}ms")
    
    print(f"\nProcessed {len(image_files)} frames")

'''