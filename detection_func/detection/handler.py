import time
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms.functional as transform
from PIL import Image
import io
import json

# Load the model and weights
start_model_loading = time.time()
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)
model.eval()
end_model_loading = time.time()

def handle(event, context):
    global start_model_loading, end_model_loading
    try:
        # Load input image from event
        image_data = event.body
        image = Image.open(io.BytesIO(image_data))
        # The image can be converted to tensor using
        img = transform.to_tensor(image)

        # Apply inference preprocessing
        preprocess = weights.transforms()
        start_preprocess = time.time()
        batch = [preprocess(img)]
        end_preprocess = time.time()

        # Run model prediction
        start_prediction = time.time()
        prediction = model(batch)[0]
        end_prediction = time.time()

        labels = [weights.meta["categories"][i] for i in prediction["labels"]]

        top_k = 10
        bboxes = []
        for i in range(top_k):
            if labels[i] == 'car':
                bboxes.append(prediction["boxes"][i].tolist())
        
        # Count and clear model loading time measurement
        model_loading_time = (end_model_loading - start_model_loading) * 1000.0
        end_model_loading = start_model_loading

        return {
            "statusCode": 200,
            "body": json.dumps({
                "bounding_boxes": bboxes,
                "model_loading_time": model_loading_time,
                "preprocessing_time": (end_preprocess - start_preprocess) * 1000.0,
                "prediction_time": (end_prediction - start_prediction) * 1000.0
            }),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Internal Server Error: {str(e)}"
        }
