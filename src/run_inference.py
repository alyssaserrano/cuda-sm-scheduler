import onnxruntime as ort
import numpy as np

# Load ONNX model
model_path = "/home/brd/cuda-sm-scheduler/models/resnet50_v1.onnx"
ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Load preprocessed image
img = np.load("/home/brd/cuda-sm-scheduler/models/imagenet_preprocessed/0001eeaf4aed83f9.jpg.npy").astype(np.float32)
img_batch = img[None, ...]  # Add batch dimension

# Run inference on GPU
outputs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: img_batch})
logits = outputs[0][0]  # shape: (1000,)

# Save logits to binary file
logits.astype(np.float32).tofile("logits.bin")
