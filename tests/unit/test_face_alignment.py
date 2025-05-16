import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock

from visiovox.modules.facial_processing.face_swapper import FaceSwapper
from visiovox.modules.scene_analysis.mediapipe_face_mesh import MediaPipeFaceMeshExtractor

# Instruções:
# Rode este teste com: pytest tests/unit/test_face_alignment.py
# Ele irá salvar imagens alinhadas em data/output/aligned_face_dummy.jpg e data/output/aligned_face_real.jpg

def gerar_landmarks_68_dummy(face_centro, escala=1.0):
    """
    Gera 68 pontos dummy, com os 5 principais em posições plausíveis para um rosto centralizado.
    """
    cx, cy = face_centro
    lms = np.zeros((68,2), dtype=np.float32)
    # Olho esquerdo (36)
    lms[36] = [cx - 30*escala, cy - 20*escala]
    # Olho direito (45)
    lms[45] = [cx + 30*escala, cy - 20*escala]
    # Nariz (30)
    lms[30] = [cx, cy]
    # Boca esquerda (48)
    lms[48] = [cx - 20*escala, cy + 30*escala]
    # Boca direita (54)
    lms[54] = [cx + 20*escala, cy + 30*escala]
    # Preencher os outros pontos com uma elipse ao redor do centro
    for i in range(68):
        if i not in [36,45,30,48,54]:
            angle = 2*np.pi*i/68
            lms[i] = [cx + 40*np.cos(angle)*escala, cy + 50*np.sin(angle)*escala]
    return lms

def test_align_face_to_template_sintetico():
    img = np.full((256,256,3), 220, dtype=np.uint8)
    cv2.circle(img, (128,128), 80, (180,160,140), -1)
    lms = gerar_landmarks_68_dummy((128,128), escala=1.0)
    cfg = MagicMock()
    res = MagicMock()
    swapper = FaceSwapper(cfg, res)
    lms5 = swapper._select_5_landmarks(lms)
    assert lms5 is not None, "Falha ao selecionar 5 landmarks"
    aligned, inv = swapper._align_face_to_template(img, lms5, (128,128))
    assert aligned is not None, "Alinhamento retornou None"
    assert aligned.shape == (128,128,3), f"Shape inesperado: {aligned.shape}"
    out_path = os.path.abspath(os.path.join('data', 'output', 'aligned_face_dummy.jpg'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, aligned)
    print(f"Imagem alinhada sintética salva em: {out_path}")

def test_alinhamento_face_real():
    # Tentar carregar imagem real em .jpeg ou .png
    img_path_jpeg = os.path.abspath(os.path.join('assets', 'demo_media', 'face1.jpeg'))
    img_path_png = os.path.abspath(os.path.join('assets', 'demo_media', 'face1.png'))
    if os.path.exists(img_path_jpeg):
        img_path = img_path_jpeg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        print(f"Nenhuma imagem encontrada em {img_path_jpeg} ou {img_path_png}. Teste pulado.")
        pytest.skip("Imagem de teste não encontrada.")
    img = cv2.imread(img_path)
    assert img is not None, f"Falha ao carregar imagem: {img_path}"

    # Extrair landmarks com MediaPipe
    extractor = MediaPipeFaceMeshExtractor()
    bbox = (0, 0, img.shape[1], img.shape[0])
    landmarks_68 = extractor.extract_landmarks(img, bbox)
    if landmarks_68 is None or landmarks_68.shape[0] < 68:
        print(f"Atenção: Foram detectados apenas {0 if landmarks_68 is None else landmarks_68.shape[0]} pontos. Teste pulado.")
        pytest.skip("Não foi possível extrair landmarks reais com MediaPipe.")

    print(f"Foram detectados {landmarks_68.shape[0]} pontos pelo extractor.")

    # Salvar imagem original com TODOS os landmarks desenhados
    img_all_points = img.copy()
    for (x, y) in landmarks_68:
        cv2.circle(img_all_points, (int(x), int(y)), 1, (0,0,255), -1)
    cv2.imwrite(os.path.join('data', 'output', 'face1_landmarks_all.jpg'), img_all_points)
    print("Imagem com todos os landmarks salva em data/output/face1_landmarks_all.jpg")

    # Salvar imagem com índices dos landmarks
    img_indexed = img.copy()
    for idx, (x, y) in enumerate(landmarks_68):
        cv2.circle(img_indexed, (int(x), int(y)), 2, (255,0,0), -1)
        cv2.putText(img_indexed, str(idx), (int(x)+2, int(y)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join('data', 'output', 'face1_landmarks_all_indexed.jpg'), img_indexed)
    print("Imagem com índices dos landmarks salva em data/output/face1_landmarks_all_indexed.jpg")

    if landmarks_68.shape[0] >= 468:
        print("AVISO: Foram detectados 468 pontos. É necessário mapear corretamente para os 5 pontos do ArcFace!")
        pytest.skip("Necessário implementar mapeamento de 468 para 5 pontos ArcFace.")

    # Se necessário, descomente para inverter o eixo Y dos landmarks
    # landmarks_68[:,1] = img.shape[0] - landmarks_68[:,1]

    # Selecionar os 5 pontos padrão ArcFace (MediaPipe FaceMesh)
    idxs = [33, 263, 1, 61, 291]
    landmarks_5 = landmarks_68[idxs]

    # Template ArcFace para 128x128
    template_112 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    scale = 128.0 / 112.0
    template_128 = template_112 * scale

    # Alinhar
    from skimage import transform as tf
    tform = tf.SimilarityTransform()
    tform.estimate(landmarks_5, template_128)
    warped = cv2.warpAffine(img, tform.params[:2], (128, 128), flags=cv2.INTER_LINEAR)

    # (Opcional) Desenhar pontos antes/depois
    img_pontos = img.copy()
    for (x, y) in landmarks_5:
        cv2.circle(img_pontos, (int(x), int(y)), 2, (0,255,0), -1)
    cv2.imwrite(os.path.join('data', 'output', 'face1_landmarks5.jpg'), img_pontos)

    warped_pontos = warped.copy()
    for (x, y) in template_128:
        cv2.circle(warped_pontos, (int(x), int(y)), 2, (0,0,255), -1)
    cv2.imwrite(os.path.join('data', 'output', 'aligned_face_real.jpg'), warped)
    cv2.imwrite(os.path.join('data', 'output', 'aligned_face_real_landmarks.jpg'), warped_pontos)
    print("Imagens salvas em data/output/aligned_face_real.jpg e data/output/aligned_face_real_landmarks.jpg")
    assert warped.shape == (128, 128, 3) 