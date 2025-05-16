import onnx
import numpy as np
import os

ONNX_PATH = os.path.join('models', 'face_swappers', 'inswapper_128.onnx')
EMAP_PATH = os.path.join('models', 'face_swappers', 'inswapper_128_emap.npy')

model = onnx.load(ONNX_PATH)
print('Inicializadores encontrados no modelo:')
for initializer in model.graph.initializer:
    print(f'- {initializer.name} | shape: {[d for d in initializer.dims]}')

found = False
for initializer in model.graph.initializer:
    if 'emap' in initializer.name.lower():
        emap = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(512, 512)
        np.save(EMAP_PATH, emap)
        print(f"emap extraída e salva em {EMAP_PATH}")
        found = True
        break
if not found:
    print("Não foi encontrada nenhuma matriz emap no modelo ONNX. Verifique o modelo.") 