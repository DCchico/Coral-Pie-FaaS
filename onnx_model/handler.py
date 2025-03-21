import onnxruntime
import numpy as np
from PIL import Image
import time

# Preprocessing function
def preprocess(img_path):
    input_shape = (1, 3, 224, 224) # (1,3,1200,1200) for ssd
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.BILINEAR) # (1200,1200) for ssd
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# Load the model
onnx_model_path = 'mobilenetv2-12.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 1  # Set the desired number of intra-op threads, delete for default, which is max number of CPU available
sess_options.inter_op_num_threads = 1  # Set the desired number of inter-op threads, default is max number of CPU available
sess = onnxruntime.InferenceSession(onnx_model_path, sess_options)

# Inference function
def inference(img_path):
    # Preprocess the input image
    input_data = preprocess(img_path)

    # Run the inference
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]

    # Measure the inference time
    start_time = time.time()
    results = sess.run(output_names, {input_name: input_data})
    end_time = time.time()
    inference_time = end_time - start_time

    # Post-process the results as needed
    # To increase the time taken for E2E, eg thresholding, filtering, visualization, extra business logic
    # for the application like uploading and sync with some DB

    return results, inference_time

# Run inferences
img_path = '000000.jpeg'
total_time = 0.0

for i in range(10):
    start_time = time.time()
    results, inference_time = inference(img_path)
    end_time = time.time()
    total_time += inference_time
    e2e_time = end_time - start_time
    print(f'Inference {i + 1}: Time taken = {inference_time} seconds')
    print(f'E2E {i + 1}: Time taken = {inference_time} seconds')
average_time = total_time / 10
print(f'Average inference time: {average_time} seconds')
