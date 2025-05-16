import os
import cv2
import numpy as np
from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager
from visiovox.modules.facial_processing.face_swapper import FaceSwapper
from visiovox.modules.scene_analysis.mediapipe_face_mesh import MediaPipeFaceMeshExtractor

# Instruções:
# Rode este script com: python scripts/test_real_face_alignment.py
# Ele irá salvar a face alinhada em data/output/aligned_face_real_sample.jpg
# e a imagem original com os 5 pontos em data/output/sample_test_image_landmarks5.jpg

def main():
    img_path = os.path.abspath(os.path.join('data', 'input', 'sample_test_image.png'))
    if not os.path.exists(img_path):
        print(f"Imagem não encontrada: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"Falha ao carregar imagem: {img_path}")
        return

    # Instanciar componentes
    cfg = ConfigManager()
    res = ResourceManager(cfg)
    extractor = MediaPipeFaceMeshExtractor()
    swapper = FaceSwapper(cfg, res)

    # Extrair landmarks (usando bbox da imagem toda)
    bbox = (0, 0, img.shape[1], img.shape[0])
    landmarks = extractor.extract_landmarks(img, bbox)
    if landmarks is None or landmarks.shape[0] < 300:
        print(f"Landmarks insuficientes extraídos: {0 if landmarks is None else landmarks.shape[0]}")
        return

    # Selecionar os 5 pontos
    landmarks_5 = swapper._select_5_landmarks(landmarks)
    if landmarks_5 is None:
        print("Falha ao selecionar 5 landmarks para alinhamento.")
        return

    # Alinhar para 128x128
    aligned, inv = swapper._align_face_to_template(img, landmarks_5, (128,128))
    if aligned is None:
        print("Falha no alinhamento facial.")
        return

    os.makedirs(os.path.join('data', 'output'), exist_ok=True)
    cv2.imwrite(os.path.join('data', 'output', 'aligned_face_real_sample.jpg'), aligned)
    print("Face alinhada salva em data/output/aligned_face_real_sample.jpg")

    # Salvar imagem original com os 5 pontos
    img_landmarks5 = img.copy()
    for (x, y) in landmarks_5:
        cv2.circle(img_landmarks5, (int(x), int(y)), 3, (0,255,0), -1)
    cv2.imwrite(os.path.join('data', 'output', 'sample_test_image_landmarks5.jpg'), img_landmarks5)
    print("Imagem original com 5 landmarks salva em data/output/sample_test_image_landmarks5.jpg")

if __name__ == "__main__":
    main() 