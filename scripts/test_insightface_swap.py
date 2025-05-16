import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import insightface
import torch

# Modelos disponíveis (nomes lógicos, igual ao VisoMaster)
scrfd_path = 'scrfd_2.5g_bnkps'
embedding_models = [
    ('antelopev2', 'antelopev2'),
    ('buffalo_l', 'buffalo_l'),
    ('buffalo_s', 'buffalo_s'),
    ('buffalo_sc', 'buffalo_sc'),
]

# Caminhos das imagens
src_path = os.path.abspath(os.path.join('data', 'input', 'source_face.png'))
tgt_path = os.path.abspath(os.path.join('data', 'input', 'target_face.png'))
if not os.path.exists(src_path) or not os.path.exists(tgt_path):
    print(f"Imagens de origem ou destino não encontradas: {src_path}, {tgt_path}")
    exit(1)
src_img = cv2.imread(src_path)
tgt_img = cv2.imread(tgt_path)
if src_img is None or tgt_img is None:
    print(f"Falha ao carregar imagens.")
    exit(1)

success = False
print(f"\nTentando carregar detector de faces: {scrfd_path}")
try:
    detector = insightface.model_zoo.get_model(scrfd_path, download=True, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    bboxes_src, kpss_src = detector.detect(src_img, max_num=1)
    bboxes_tgt, kpss_tgt = detector.detect(tgt_img, max_num=1)
    if bboxes_src is None or len(bboxes_src) == 0 or bboxes_tgt is None or len(bboxes_tgt) == 0:
        print(f"Nenhuma face detectada usando {scrfd_path}!")
    else:
        bbox_src = bboxes_src[0]
        kps_src = kpss_src[0]
        bbox_tgt = bboxes_tgt[0]
        kps_tgt = kpss_tgt[0]
        from insightface.utils.face_align import norm_crop
        aligned_src = norm_crop(src_img, kps_src)
        for emb_name, emb_logical in embedding_models:
            print(f"\nTestando embedding model: {emb_name} ({emb_logical})")
            try:
                arcface = insightface.model_zoo.get_model(emb_logical, download=True, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                embedding_src = arcface.get(aligned_src)
                print("Carregando modelo de swap do insightface...")
                swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("Executando swap facial com insightface...")
                result_img = swapper.get(tgt_img, kps_tgt, embedding_src, paste_back=True)
                os.makedirs(os.path.join('data', 'output'), exist_ok=True)
                out_path = os.path.join('data', 'output', f'insightface_swap_result_scrfd_{emb_name}.jpg')
                cv2.imwrite(out_path, result_img)
                print(f"Swap facial concluído com SCRFD + {emb_name}! Resultado salvo em {out_path}")
                success = True
            except Exception as e:
                print(f"Erro ao tentar usar embedding {emb_name}: {e}")
except Exception as e:
    print(f"Erro ao tentar usar detector SCRFD: {e}")

if not success:
    print("Nenhuma das combinações testadas funcionou com o insightface. Use modelos oficiais e compatíveis do projeto InsightFace.") 