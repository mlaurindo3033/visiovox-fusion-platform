from ultralytics import SAM
import numpy as np
from PIL import Image

MODEL_MAP = {
    'Rápido (Tiny)': 'model_assets/sam2/sam2.1_t.pt',
    'Equilibrado (Small)': 'model_assets/sam2/sam2.1_s.pt',
    'Qualidade Máxima (Large)': 'model_assets/sam2/sam2.1_l.pt',
}

class SAM2FaceSegmenter:
    def __init__(self, model_name: str):
        model_path = MODEL_MAP.get(model_name, MODEL_MAP['Rápido (Tiny)'])
        self.model = SAM(model_path)
        self.model.eval()

    def segment_face(self, image: np.ndarray, return_info: bool = False):
        """
        Recebe uma imagem (np.ndarray ou PIL), retorna máscara binária do rosto.
        Se return_info=True, retorna também o número de máscaras e a área da maior.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        results = self.model.predict(pil_img)
        masks = [np.array(mask.data[0].cpu()) for mask in results[0].masks]
        if not masks:
            mask = np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
            n_masks = 0
            max_area = 0
        else:
            mask = max(masks, key=lambda m: np.sum(m))
            max_area = int(np.sum(mask > 0.5))
            mask = (mask > 0.5).astype(np.uint8) * 255
            n_masks = len(masks)
        if return_info:
            return mask, n_masks, max_area
        return mask

# Exemplo de uso:
# segmenter = SAM2FaceSegmenter('Rápido (Tiny)')
# mask = segmenter.segment_face(image) 