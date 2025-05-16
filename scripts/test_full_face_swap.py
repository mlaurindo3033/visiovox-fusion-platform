import os
import cv2
import numpy as np
from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager
from visiovox.modules.facial_processing.face_swapper import FaceSwapper
from visiovox.modules.scene_analysis.mediapipe_face_mesh import MediaPipeFaceMeshExtractor
from visiovox.modules.scene_analysis.face_recogniser import FaceRecogniser
from visiovox.modules.facial_processing.face_detector_yoloface8 import YoloFace8Detector
import logging
import torch

# Instruções:
# Rode este script com: python scripts/test_full_face_swap.py
# Ele irá salvar o resultado do swap em data/output/full_face_swap_result.jpg

logging.basicConfig(level=logging.DEBUG)

def main():
    # Definir device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('AVISO: CUDA não disponível, usando CPU. O modelo FP16 pode não funcionar corretamente!')

    # Caminhos das imagens
    src_path = os.path.abspath(os.path.join('data', 'input', 'source_face.png'))
    tgt_path = os.path.abspath(os.path.join('data', 'input', 'target_face.png'))
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        print(f"Imagens de origem ou destino não encontradas: {src_path}, {tgt_path}")
        return
    src_img = cv2.imread(src_path)
    tgt_img = cv2.imread(tgt_path)
    if src_img is None or tgt_img is None:
        print(f"Falha ao carregar imagens.")
        return

    # Instanciar componentes
    cfg = ConfigManager()
    res = ResourceManager(cfg)
    extractor = MediaPipeFaceMeshExtractor()
    recogniser = FaceRecogniser(cfg, res)
    recogniser.load_model('simswap_arcface_visomaster')
    swapper = FaceSwapper(cfg, res, device=device)
    ok = swapper.load_model('inswapper_128_fp16')
    print('Swapper carregado?', ok)
    detector = YoloFace8Detector(conf_thresh=0.2)

    # --- Origem: detectar rosto, extrair landmarks e embedding ---
    print("Detectando face de origem com YoloFace8...")
    bbox_src = detector.detect_face(src_img)
    if bbox_src is None:
        print("Nenhuma face detectada na imagem de origem!")
        return
    # Salvar imagem de debug com bounding box
    debug_img_src_bbox = src_img.copy()
    x, y, w, h = bbox_src
    cv2.rectangle(debug_img_src_bbox, (x, y), (x+w, y+h), (0,255,255), 2)
    cv2.imwrite('data/output/debug_bbox_src.jpg', debug_img_src_bbox)
    src_landmarks = extractor.extract_landmarks(src_img, bbox_src)
    if src_landmarks is None or src_landmarks.shape[0] < 300:
        print(f"Landmarks insuficientes na origem: {0 if src_landmarks is None else src_landmarks.shape[0]}")
        return
    src_landmarks_5 = swapper._select_5_landmarks(src_landmarks)
    debug_img_src = src_img.copy()
    if src_landmarks_5 is not None:
        for (x, y) in src_landmarks_5.astype(int):
            cv2.circle(debug_img_src, (x, y), 3, (0,255,0), -1)
    if src_landmarks_5 is None:
        print("Falha ao selecionar 5 landmarks na origem.")
        return
    # Alinhar para 128x128 para embedding
    aligned_src, _ = swapper._align_face_to_template(src_img, src_landmarks_5, (128,128))
    if aligned_src is None:
        print("Falha ao alinhar face de origem para embedding.")
        return
    # O FaceRecogniser espera imagem alinhada (128x128, RGB)
    src_embedding = recogniser.get_face_embedding(src_img, src_landmarks_5)
    if src_embedding is None:
        print("Falha ao extrair embedding da origem.")
        return

    # --- Destino: detectar rosto, extrair landmarks ---
    print("Detectando face de destino com YoloFace8...")
    bbox_tgt = detector.detect_face(tgt_img)
    if bbox_tgt is None:
        print("Nenhuma face detectada na imagem de destino!")
        return
    tgt_landmarks = extractor.extract_landmarks(tgt_img, bbox_tgt)
    if tgt_landmarks is None or tgt_landmarks.shape[0] < 300:
        print(f"Landmarks insuficientes no destino: {0 if tgt_landmarks is None else tgt_landmarks.shape[0]}")
        return
    tgt_landmarks_5 = swapper._select_5_landmarks(tgt_landmarks)
    debug_img_tgt = tgt_img.copy()
    if tgt_landmarks_5 is not None:
        for (x, y) in tgt_landmarks_5.astype(int):
            cv2.circle(debug_img_tgt, (x, y), 3, (0,0,255), -1)
        cv2.imwrite('data/output/debug_landmarks_tgt.jpg', debug_img_tgt)

    # --- Swap ---
    print("Executando swap facial...")
    result_img = swapper.swap_face(src_embedding, tgt_img, tgt_landmarks)
    if result_img is None:
        print("Swap facial falhou.")
        return

    os.makedirs(os.path.join('data', 'output'), exist_ok=True)
    out_path = os.path.join('data', 'output', 'full_face_swap_result.jpg')
    cv2.imwrite(out_path, result_img)
    print(f"Swap facial concluído! Resultado salvo em {out_path}")

if __name__ == "__main__":
    main() 