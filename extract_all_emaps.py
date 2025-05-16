import onnx
import numpy as np
import os

def extract_emap_from_onnx(onnx_path, npy_path):
    model = onnx.load(onnx_path)
    found = False
    for initializer in model.graph.initializer:
        # Procura nomes comuns para emap
        if any(key in initializer.name.lower() for key in ['emap', 'arcface', 'mapper']):
            # Tenta shape padrão (512, 512) ou variantes
            shape = tuple(initializer.dims)
            if len(shape) == 2 and shape[0] == shape[1]:
                emap = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(shape)
                np.save(npy_path, emap)
                print(f"emap extraída de {onnx_path} e salva em {npy_path} (shape={shape})")
                found = True
                break
    if not found:
        print(f"[AVISO] Nenhuma matriz emap encontrada em {onnx_path}.")

if __name__ == "__main__":
    base_dir = os.path.join('models', 'face_swappers')
    for fname in os.listdir(base_dir):
        if fname.endswith('.onnx'):
            onnx_path = os.path.join(base_dir, fname)
            npy_path = os.path.splitext(onnx_path)[0] + '_emap.npy'
            extract_emap_from_onnx(onnx_path, npy_path) 