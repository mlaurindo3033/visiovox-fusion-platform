import cv2
import numpy as np
import mediapipe as mp

class MediaPipeFaceMeshExtractor:
    """
    Fallback para extração de landmarks faciais usando MediaPipe FaceMesh.
    """
    def __init__(self, static_image: bool = True, max_faces: int = 1, min_confidence: float = 0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_confidence)

    def extract_landmarks(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extrai landmarks faciais usando MediaPipe FaceMesh dentro de uma bbox.
        Args:
            image (np.ndarray): Imagem BGR completa.
            bbox (tuple): Bounding box (x, y, w, h).
        Returns:
            np.ndarray: Array de pontos (N,2) ou None se não encontrar.
        """
        x, y, w, h = bbox
        # Converter para RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        h_img, w_img = image.shape[:2]
        # Pegar landmarks da primeira face detectada
        lm = results.multi_face_landmarks[0].landmark
        pts = []
        x2, y2 = x + w, y + h
        for l in lm:
            px = int(l.x * w_img)
            py = int(l.y * h_img)
            # Filtrar somente pontos dentro da bbox
            if x <= px <= x2 and y <= py <= y2:
                pts.append([px, py])
        if not pts:
            return None
        return np.array(pts, dtype=np.int32) 