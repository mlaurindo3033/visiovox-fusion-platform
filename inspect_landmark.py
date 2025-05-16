import onnxruntime

for model_name in ['2dfan4', 'fan_68_5']:
    path = f"models/face_landmarkers/{model_name}.onnx"
    print(f"Inspecting {model_name}: {path}")
    sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
    inputs = [(i.name, i.shape) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape) for o in sess.get_outputs()]
    print("  Inputs: ", inputs)
    print("  Outputs:", outputs) 