import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
import face_recognition
from torchvision.transforms import v2
from skimage import transform as trans
from skimage.exposure import match_histograms
import mediapipe as mp

# ===================== CONFIGURAÇÕES =====================
# Caminhos dos arquivos (ajuste conforme necessário)
CSCS_ONNX_PATH = 'model_assets/cscs_256.onnx'  # Caminho para o modelo CSCS
ARCFACE_ONNX_PATH = 'model_assets/cscs_arcface_model.onnx'  # Caminho para o modelo ArcFace usado pelo CSCS
IMG_TARGET_PATH = 'testdata/target_face.jpg'  # Imagem de destino (rosto a ser trocado)
IMG_SOURCE_PATH = 'testdata/source_face.jpg'  # Imagem de origem (rosto a ser inserido)
CSCS_ID_ADAPTER_ONNX_PATH = 'model_assets/cscs_id_adapter.onnx'  # Caminho para o modelo ID Adapter usado pelo CSCS

# ===================== FUNÇÕES AUXILIARES =====================
def get_ffhq_template():
    # Template FFHQ (padrão para CSCS)
    return np.array([
        [89.3095, 114.617],
        [169.3095, 114.617],
        [129.3095, 162.617],
        [99.3095, 202.617],
        [159.3095, 202.617],
    ], dtype=np.float32)

def extract_5pts_from_landmarks(landmarks):
    # Extrai os 5 pontos principais do dlib/face_recognition
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    nose = np.mean(landmarks['nose_tip'], axis=0)
    mouth_left = landmarks['top_lip'][0]
    mouth_right = landmarks['top_lip'][6]
    return np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)

def detect_face_and_kps(img):
    # Detecta a face e retorna os 5 keypoints principais
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        raise RuntimeError('Nenhuma face detectada na imagem.')
    face_landmarks_list = face_recognition.face_landmarks(img, face_locations)
    if not face_landmarks_list:
        raise RuntimeError('Nenhum landmark detectado.')
    kps_5 = extract_5pts_from_landmarks(face_landmarks_list[0])
    return kps_5

def align_face_torch(img, kps_5, dst_pts, out_size=256):
    # img: numpy array (H, W, C) RGB
    # Converte para tensor e normaliza para [0,1]
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)  # (C, H, W)
    tform = trans.SimilarityTransform()
    tform.estimate(kps_5, dst_pts)
    # Aplica affine
    affine_img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]), tform.scale, 0, center=(0,0), interpolation=v2.InterpolationMode.BILINEAR)
    # Crop
    affine_img = v2.functional.crop(affine_img, 0, 0, out_size, out_size)
    return affine_img, tform

def normalize_img_torch(img):
    # img: tensor (C, H, W) em [0,1]
    return v2.functional.normalize(img, [0.5,0.5,0.5], [0.5,0.5,0.5])

def denormalize_img_torch(img):
    # img: tensor (C, H, W) em [-1,1]
    return img * 0.5 + 0.5

def calc_swapper_latent_cscs(embedding):
    # Simples reshape para (1, -1)
    latent = embedding.reshape(1, -1)
    return latent

def get_arcface_embedding(img, kps_5, arcface_onnx_path):
    # Alinha a face para 112x112 usando template ArcFace
    arcface_template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(kps_5, arcface_template)
    M = tform.params[0:2]
    aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
    aligned = aligned.astype(np.float32)
    # Normalização ArcFace
    aligned = (aligned - 127.5) / 127.5
    aligned = np.transpose(aligned, (2, 0, 1))[None, ...]  # 1x3x112x112
    # Inferência ONNX
    sess = ort.InferenceSession(arcface_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    embedding = sess.run([output_name], {input_name: aligned})[0]
    embedding = embedding.flatten()
    # Normalização L2
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def get_id_adapter_embedding(img, id_adapter_onnx_path):
    # img: tensor (1, 3, 112, 112) em [-1,1]
    img_np = img.cpu().numpy()  # 1x3x112x112
    sess = ort.InferenceSession(id_adapter_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    embedding = sess.run([output_name], {input_name: img_np.astype(np.float32)})[0]
    embedding = embedding.flatten()
    # Normalização L2
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def get_face_mask(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for idx in range(468):
                    x = int(face_landmarks.landmark[idx].x * image.shape[1])
                    y = int(face_landmarks.landmark[idx].y * image.shape[0])
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 1)
        return mask

def postprocess_face_swap(
    swapped_img, img_target, feather=11, blend_w=0.3, d=9, sigmaColor=75, sigmaSpace=75
):
    mask = get_face_mask(swapped_img)
    mask_blur = cv2.GaussianBlur(mask, (feather, feather), 0)
    matched_region = match_histograms(swapped_img, img_target, channel_axis=-1)
    blended = swapped_img.copy()
    for c in range(3):
        blended[..., c] = (
            swapped_img[..., c].astype(np.float32) * (1 - mask_blur * blend_w) +
            matched_region[..., c].astype(np.float32) * (mask_blur * blend_w)
        )
    smoothed = cv2.bilateralFilter(blended.astype('uint8'), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    final = blended.copy()
    for c in range(3):
        final[..., c] = (
            blended[..., c].astype(np.float32) * (1 - mask_blur) +
            smoothed[..., c].astype(np.float32) * mask_blur
        )
    final = np.clip(final, 0, 255).astype('uint8')
    return final

# ===================== PIPELINE DE TESTE =====================
print('--- Carregando modelos ONNX ---')
sess_cscs = ort.InferenceSession(CSCS_ONNX_PATH, providers=['CPUExecutionProvider'])
sess_arcface = ort.InferenceSession(ARCFACE_ONNX_PATH, providers=['CPUExecutionProvider'])

print('--- Lendo imagens ---')
img_target = cv2.imread(IMG_TARGET_PATH)
img_source = cv2.imread(IMG_SOURCE_PATH)
if img_target is None or img_source is None:
    raise FileNotFoundError('Imagem de destino ou origem não encontrada.')
img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
print(f'Imagem de destino: {img_target.shape}, Imagem de origem: {img_source.shape}')

# ========== Detecção automática de keypoints ==========
print('--- Detectando keypoints automaticamente ---')
kps_5_target = detect_face_and_kps(img_target)
kps_5_source = detect_face_and_kps(img_source)
print(f'Keypoints destino: {kps_5_target}')
print(f'Keypoints origem: {kps_5_source}')

# ========== Alinhamento da face de destino ==========
print('--- Alinhando face de destino ---')
dst_pts = get_ffhq_template()
aligned_target_face, tform = align_face_torch(img_target, kps_5_target, dst_pts, out_size=256)
print(f'Alinhada: {aligned_target_face.shape}')

# ========== Obtenção do embedding de origem ==========
print('--- Calculando embedding ArcFace da origem ---')
embedding_source = get_arcface_embedding(img_source, kps_5_source, ARCFACE_ONNX_PATH)
print(f'Embedding origem: shape={embedding_source.shape}, min={embedding_source.min()}, max={embedding_source.max()}')

# ========== Calculando embedding ID Adapter da origem ==========
print('--- Calculando embedding ID Adapter da origem ---')
aligned_source_face, _ = align_face_torch(img_source, kps_5_source, dst_pts, out_size=256)
# Redimensiona para 112x112 (mantendo C, H, W)
input_id_adapter = torch.nn.functional.interpolate(aligned_source_face.unsqueeze(0), size=(112,112), mode='bilinear', align_corners=False)
input_id_adapter = normalize_img_torch(input_id_adapter[0]).unsqueeze(0)  # (1, 3, 112, 112)
embedding_id = get_id_adapter_embedding(input_id_adapter, CSCS_ID_ADAPTER_ONNX_PATH)
print(f'Embedding ID Adapter: shape={embedding_id.shape}, min={embedding_id.min()}, max={embedding_id.max()}')

# ========== Somando embeddings ArcFace + ID Adapter ==========
print('--- Somando embeddings ArcFace + ID Adapter ---')
embedding_sum = embedding_source + embedding_id
embedding_sum = embedding_sum / np.linalg.norm(embedding_sum)

# ========== Cálculo do latente CSCS ==========
print('--- Calculando latente CSCS ---')
latent = calc_swapper_latent_cscs(embedding_sum)
print(f'Latente: shape={latent.shape}, min={latent.min()}, max={latent.max()}')

# ========== Preparação da entrada para o CSCS ==========
print('--- Preparando entrada para CSCS ---')
input_face = normalize_img_torch(aligned_target_face)
print(f'Input face normalizada: shape={input_face.shape}, min={input_face.min()}, max={input_face.max()}')
input_face = input_face.unsqueeze(0).numpy()  # 1x3x256x256

# ========== Inferência CSCS ==========
print('--- Executando inferência CSCS ---')
input_name_img = sess_cscs.get_inputs()[0].name
input_name_latent = sess_cscs.get_inputs()[1].name
output_names = [o.name for o in sess_cscs.get_outputs()]
print(f'Nomes de entrada: {input_name_img}, {input_name_latent}')
print(f'Nomes de saída: {output_names}')
outputs = sess_cscs.run(None, {input_name_img: input_face.astype(np.float32), input_name_latent: latent.astype(np.float32)})
print(f'Número de outputs: {len(outputs)}')
for i, out in enumerate(outputs):
    print(f'Output[{i}]: shape={out.shape}, dtype={out.dtype}, min={out.min()}, max={out.max()}, mean={out.mean()}')

# ========== Pós-processamento e salvamento ==========
print('--- Pós-processando e salvando resultado ---')
output_img = outputs[0]  # Assumindo que o primeiro é a imagem trocada
output_img = torch.from_numpy(output_img)
output_img = torch.squeeze(output_img)
output_img = denormalize_img_torch(output_img)
output_img = torch.clamp(output_img, 0, 1)
output_img = (output_img * 255).byte()
output_img = output_img.permute(1, 2, 0).cpu().numpy()  # HWC
cv2.imwrite('swapped_face_cscs.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
print('Imagem resultante salva como swapped_face_cscs.png')

# Pós-processamento modular após o swap
final = postprocess_face_swap(
    output_img, img_target,
    feather=11, blend_w=0.3, d=9, sigmaColor=75, sigmaSpace=75
)
cv2.imwrite('swap_final_best.png', cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
print('Imagem final pós-processamento modular salva como swap_final_best.png') 