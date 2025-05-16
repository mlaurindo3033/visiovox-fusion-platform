import cv2
import numpy as np
from typing import List, Tuple, Optional

def draw_detections_and_landmarks(
    image_array: np.ndarray,
    detections_xywh: List[Tuple[int, int, int, int]],
    all_landmarks: Optional[List[Optional[np.ndarray]]] = None,
    bbox_color: Tuple[int, int, int] = (0, 255, 0),      # Verde para bboxes
    landmark_color: Tuple[int, int, int] = (0, 0, 255), # Vermelho para landmarks
    bbox_thickness: int = 2,
    landmark_radius: int = 2
) -> None:
    """
    Desenha bounding boxes de detecção e, opcionalmente, landmarks faciais na imagem.

    Args:
        image_array (np.ndarray): A imagem (array NumPy BGR) onde desenhar.
                                    Esta imagem é modificada no local.
        detections_xywh (List[Tuple[int, int, int, int]]): Uma lista de bounding boxes.
                                      Cada bbox é uma tupla (x, y, w, h).
        all_landmarks (Optional[List[Optional[np.ndarray]]]): Uma lista de arrays de landmarks.
                                      Cada item da lista corresponde a uma detecção em detections_xywh.
                                      Um array de landmarks é esperado como (N, 2) para N pontos (x,y).
                                      Se None, ou um item específico da lista for None, os landmarks
                                      para essa detecção são ignorados.
        bbox_color (Tuple[int, int, int]): Cor para as bounding boxes (BGR).
        landmark_color (Tuple[int, int, int]): Cor para os landmarks (BGR).
        bbox_thickness (int): Espessura das linhas da bounding box.
        landmark_radius (int): Raio dos círculos dos landmarks.
    """
    num_detections = len(detections_xywh)

    for i in range(num_detections):
        x, y, w, h = detections_xywh[i]
        # Desenhar a bounding box
        cv2.rectangle(image_array, (x, y), (x + w, y + h), bbox_color, bbox_thickness)

        # Desenhar landmarks se disponíveis para esta detecção
        if all_landmarks and i < len(all_landmarks):
            landmarks_for_face = all_landmarks[i]
            if landmarks_for_face is not None:
                for (lx, ly) in landmarks_for_face:
                    # Assegurar que lx e ly são inteiros para cv2.circle
                    cv2.circle(image_array, (int(round(lx)), int(round(ly))), landmark_radius, landmark_color, -1) # -1 para preencher o círculo 