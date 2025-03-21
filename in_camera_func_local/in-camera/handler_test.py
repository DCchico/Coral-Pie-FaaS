from handler import handle
import logging
# Test your handler here

# To disable testing, you can set the build_arg `TEST_ENABLED=false` on the CLI or in your stack.yml
# https://docs.openfaas.com/reference/yaml/#function-build-args-build-args
logging.basicConfig(level=logging.INFO)

def test_handle():
    input_data = {
        "camera_id": 0,
        "frame_id": 0,
        "image": "nothing needed for now",  # Replace with actual image data
        "bboxes": [[183.3546, 233.2469, 267.5851, 296.2590, 100]]
    }

    result = handle(input_data, None)
    print(result)
    return result

