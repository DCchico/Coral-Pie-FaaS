import time
import os
import json
import glob
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms.functional as transform
from PIL import Image
import io

# Path constants
IMAGE_DIR = "/usr/local/src/Coral-Pie-FaaS/camera_stream_data"
OUTPUT_FILE = "bounding_boxes.json"

def process_image(image_path):
    """Process a single image and return its bounding boxes"""
    try:
        # Read the image file
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Load as PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to tensor
        img = transform.to_tensor(image)

        # Apply inference preprocessing
        preprocess = weights.transforms()
        batch = [preprocess(img)]

        # Run model prediction
        prediction = model(batch)[0]
        
        # Extract bounding boxes and scores
        boxes = prediction["boxes"].detach().cpu().numpy().tolist()
        scores = prediction["scores"].detach().cpu().numpy().tolist()
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        
        # Filter by confidence threshold
        valid_boxes = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > 0.3:  # Lower threshold for more detections
                valid_boxes.append({
                    "box": box,
                    "score": score,
                    "label": labels[i]
                })
        
        return valid_boxes
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []

def main():
    # Get all JPEG files
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image and collect bounding boxes
    results = {}
    for i, image_file in enumerate(image_files):
        frame_id = os.path.basename(image_file).split('.')[0]
        print(f"Processing image {i+1}/{len(image_files)}: {frame_id}")
        
        boxes = process_image(image_file)
        
        # Convert to format expected by your Go client
        go_format_boxes = []
        for detection in boxes:
            # Format: [x1, y1, x2, y2, score]
            go_format_boxes.append(detection["box"] + [detection["score"]])
        
        results[frame_id] = go_format_boxes
    
    # Save results to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Print some statistics
    total_detections = sum(len(boxes) for boxes in results.values())
    avg_detections = total_detections / len(results) if results else 0
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_detections:.2f}")

if __name__ == "__main__":
    # Load the model and weights (outside the main function for global access)
    print("Loading model...")
    start_model_loading = time.time()
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.3)
    model.eval()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    end_model_loading = time.time()
    print(f"Model loaded in {(end_model_loading - start_model_loading):.2f} seconds")
    
    # Run the main processing
    main() 