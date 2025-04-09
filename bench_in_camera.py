import base64
import cv2
import numpy as np
import time
import statistics
import os

# Configuration
image_path = "/usr/local/src/Coral-Pie-FaaS/in_camera_func/in-camera/coldstart.jpeg"
iterations = 100  # How many times to run each test

# Utility function for timing
def measure_time(func, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.time()
        result = func()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds
    return result, times

# 1. Read the file and prepare base64 encoding
with open(image_path, "rb") as image_file:
    image_binary = image_file.read()
    encoded_image = base64.b64encode(image_binary).decode('utf-8')

print(f"Image: {image_path}")
print(f"Size: {len(image_binary)/1024:.2f} KB ({len(encoded_image)/1024:.2f} KB as base64)")

# 2. Test traditional imread for comparison
def test_imread():
    return cv2.imread(image_path)

orig_image, imread_times = measure_time(test_imread, iterations)
print("\n--- Traditional cv2.imread ---")
print(f"Mean: {statistics.mean(imread_times):.2f} ms")
print(f"Min: {min(imread_times):.2f} ms")
print(f"Max: {max(imread_times):.2f} ms")
print(f"Median: {statistics.median(imread_times):.2f} ms")
print(f"Std Dev: {statistics.stdev(imread_times):.2f} ms")

# 3. Test base64 decoding (step 1)
def test_base64_decode():
    return base64.b64decode(encoded_image)

decoded_image, base64_times = measure_time(test_base64_decode, iterations)
print("\n--- Step 1: base64.b64decode ---")
print(f"Mean: {statistics.mean(base64_times):.2f} ms")
print(f"Min: {min(base64_times):.2f} ms")
print(f"Max: {max(base64_times):.2f} ms")
print(f"Median: {statistics.median(base64_times):.2f} ms")
print(f"Std Dev: {statistics.stdev(base64_times):.2f} ms")

# 4. Test numpy frombuffer (step 2)
def test_np_frombuffer():
    return np.frombuffer(decoded_image, np.uint8)

nparr, numpy_times = measure_time(test_np_frombuffer, iterations)
print("\n--- Step 2: np.frombuffer ---")
print(f"Mean: {statistics.mean(numpy_times):.2f} ms")
print(f"Min: {min(numpy_times):.2f} ms")
print(f"Max: {max(numpy_times):.2f} ms")
print(f"Median: {statistics.median(numpy_times):.2f} ms")
print(f"Std Dev: {statistics.stdev(numpy_times):.2f} ms")

# 5. Test cv2 imdecode (step 3)
def test_cv2_imdecode():
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

image, imdecode_times = measure_time(test_cv2_imdecode, iterations)
print("\n--- Step 3: cv2.imdecode ---")
print(f"Mean: {statistics.mean(imdecode_times):.2f} ms")
print(f"Min: {min(imdecode_times):.2f} ms")
print(f"Max: {max(imdecode_times):.2f} ms")
print(f"Median: {statistics.median(imdecode_times):.2f} ms")
print(f"Std Dev: {statistics.stdev(imdecode_times):.2f} ms")

# 6. Test combined serverless pipeline
def test_combined_pipeline():
    decoded = base64.b64decode(encoded_image)
    nparr = np.frombuffer(decoded, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

_, combined_times = measure_time(test_combined_pipeline, iterations)
print("\n--- Complete Serverless Pipeline ---")
print(f"Mean: {statistics.mean(combined_times):.2f} ms")
print(f"Min: {min(combined_times):.2f} ms")
print(f"Max: {max(combined_times):.2f} ms")
print(f"Median: {statistics.median(combined_times):.2f} ms")
print(f"Std Dev: {statistics.stdev(combined_times):.2f} ms")

# 7. Image info
if image is not None:
    print(f"\nImage Shape: {image.shape}")
    print(f"Image Type: {image.dtype}")
else:
    print("\nFailed to decode image!")

# 8. Print summary comparison
print("\n--- Summary (Median Times) ---")
print(f"Traditional cv2.imread: {statistics.median(imread_times):.2f} ms")
print(f"Step 1 (base64 decode): {statistics.median(base64_times):.2f} ms")
print(f"Step 2 (np.frombuffer): {statistics.median(numpy_times):.2f} ms")
print(f"Step 3 (cv2.imdecode): {statistics.median(imdecode_times):.2f} ms")
print(f"Complete serverless pipeline: {statistics.median(combined_times):.2f} ms")
print(f"Sum of individual steps: {statistics.median(base64_times) + statistics.median(numpy_times) + statistics.median(imdecode_times):.2f} ms")