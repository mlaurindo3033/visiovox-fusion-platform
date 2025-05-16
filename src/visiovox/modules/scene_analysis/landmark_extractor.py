import logging
import os
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import onnxruntime

from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager

class LandmarkExtractor:
    """
    Extracts facial landmarks from a given face region using an ONNX model.
    """

    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager):
        """
        Initializes the LandmarkExtractor.

        Args:
            config_manager (ConfigManager): Instance of the configuration manager.
            resource_manager (ResourceManager): Instance of the resource manager.
        """
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.landmark_model: Optional[onnxruntime.InferenceSession] = None
        self.model_name_loaded: Optional[str] = None
        
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._input_shape: Optional[Tuple[int, int]] = None # (height, width) expected by the ONNX model
        
        self.logger.info("LandmarkExtractor initialized.")

    def _get_model_config(self, model_name: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Helper to get model-specific config from the model_manifest via ConfigManager."""
        # Assuming model catalog is loaded into 'model_catalog' key by ConfigManager
        # And structure is model_catalog.<model_type>.<model_name>.<key>
        # For landmarkers, model_type is "face_landmarker"
        config_key = f"model_catalog.face_landmarkers.{model_name}.{key}"
        value = self.config_manager.get(config_key, default)
        if value is None and default is None: # Key must exist if no default
             self.logger.warning(f"Configuration key '{config_key}' not found for landmark model '{model_name}'.")
        return value

    def load_model(self, model_name: str) -> bool:
        """
        Loads the specified landmark extraction model.
        The model type is assumed to be 'face_landmarker'.
        """
        if self.landmark_model and self.model_name_loaded == model_name:
            self.logger.info(f"Landmark model '{model_name}' is already loaded.")
            return True

        self.logger.info(f"Attempting to load landmark model: {model_name}")
        # Corrigido: Usar model_type="face_landmarker" (singular)
        model_path_or_info = self.resource_manager.load_model(model_name=model_name, model_type="face_landmarker")

        if not model_path_or_info:
            self.logger.error(f"Failed to get model path for landmark model '{model_name}' from ResourceManager.")
            return False

        # Determinar o caminho absoluto do modelo
        if isinstance(model_path_or_info, str):
            model_path_manifest_relative = model_path_or_info
        elif isinstance(model_path_or_info, dict) and 'path_local_cache' in model_path_or_info:
            model_path_manifest_relative = model_path_or_info['path_local_cache']
        else:
            self.logger.error(f"Invalid model info received from ResourceManager for '{model_name}'. Expected str or dict with 'path_local_cache'.")
            return False

        try:
            project_root = self.config_manager.get_project_root()
            absolute_model_path = os.path.join(project_root, model_path_manifest_relative)

            if not os.path.exists(absolute_model_path):
                self.logger.error(f"ONNX landmark model file not found at resolved path: {absolute_model_path}")
                # Tentar baixar/verificar novamente
                self.logger.info(f"Attempting to re-verify/download '{model_name}' via ResourceManager.")
                model_path_or_info_retry = self.resource_manager.load_model(
                    model_name=model_name, 
                    model_type="face_landmarker", 
                    force_download_check=True 
                )
                if not model_path_or_info_retry or not os.path.exists(os.path.join(project_root, model_path_or_info_retry if isinstance(model_path_or_info_retry, str) else model_path_or_info_retry.get('path_local_cache', ''))):
                    self.logger.error(f"Failed to find or download landmark model '{model_name}' after retry.")
                    return False
                # Atualiza o absolute_model_path
                if isinstance(model_path_or_info_retry, str):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry)
                elif isinstance(model_path_or_info_retry, dict):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry['path_local_cache'])

            self.logger.debug(f"Loading ONNX landmark session from: {absolute_model_path}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.landmark_model = onnxruntime.InferenceSession(absolute_model_path, providers=providers)
            self.model_name_loaded = model_name
            self.logger.info(f"Landmark model '{model_name}' loaded successfully.")

            model_details = self.resource_manager.get_model_details_from_manifest(model_name, "face_landmarker")
            if model_details and 'input_size_hw' in model_details:
                self._input_shape = tuple(model_details['input_size_hw'])
                self.logger.info(f"Model '{model_name}' input size set to: {self._input_shape}")
            else:
                default_input_size = self.config_manager.get_config(f"models.face_landmarker.{model_name}.input_size_hw", default=[256, 256])
                self._input_shape = tuple(default_input_size)
                self.logger.warning(f"Input size for '{model_name}' not found in manifest, using default/config: {self._input_shape}")

            self._input_name = self.landmark_model.get_inputs()[0].name
            model_actual_input_shape = self.landmark_model.get_inputs()[0].shape # e.g. [1, 3, 256, 256]
            
            self._output_names = [output.name for output in self.landmark_model.get_outputs()]
            
            self.logger.info(f"Model '{model_name}' I/O: Input='{self._input_name}', Output='{self._output_names[0] if self._output_names else 'Unknown'}', Expected Shape HW={self._input_shape}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX landmark model '{model_name}' from {absolute_model_path if 'absolute_model_path' in locals() else model_path_manifest_relative}: {e}", exc_info=True)
            self.landmark_model = None
            return False

    def _preprocess_face_roi(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Crops the face ROI, resizes it for the landmark model, and preprocesses.

        Args:
            image (np.ndarray): The full image.
            face_bbox (Tuple[int, int, int, int]): Bounding box of the face (x, y, w, h).

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: 
                The preprocessed face image batch and the inverse affine transform.
                Returns None if preprocessing fails.
        """
        if self._input_shape is None:
            self.logger.error("Model input shape not configured for landmark extraction.")
            return None
            
        target_h, target_w = self._input_shape
        x, y, w, h = face_bbox

        # Ensure ROI coordinates are within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        # Adjust w, h if x+w or y+h are out of bounds
        w = min(w, img_w - x) 
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            self.logger.warning(f"Invalid face_bbox resulted in zero or negative width/height: ({x},{y},{w},{h}). Cannot extract landmarks.")
            return None

        # Compute center-based rotation+scale transform (no rotation here) akin to facefusion's warp
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        # scale so that the larger side maps to the model input size, with a 1.5 padding factor
        scale = target_w / (max(w, h) * 1.5)
        M = cv2.getRotationMatrix2D((center_x, center_y), 0.0, scale)
        # Warp the full image to the model input, cropping around the bbox center
        warped_roi = cv2.warpAffine(image, M, (target_w, target_h), flags=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        normalized_face_roi = warped_roi.astype(np.float32) / 255.0
        # HWC to CHW and add batch dimension
        chw_face_roi = np.transpose(normalized_face_roi, (2, 0, 1))
        batch_face_roi = np.expand_dims(chw_face_roi, axis=0)

        # Compute inverse affine to map predicted landmarks back to original image coords
        M_inv = cv2.invertAffineTransform(M)
        return batch_face_roi, M_inv

    def _postprocess_landmarks(self, outputs: List[np.ndarray], 
                               M_inv: np.ndarray) -> Optional[np.ndarray]:
        """
        Postprocesses the raw output from the landmark model.
        This version assumes the model outputs landmark coordinates scaled to a feature map (e.g., 64x64),
        based on 2dfan4's heatmap output size and insights from visomaster code.
        """
        if not outputs or outputs[0] is None:
            self.logger.warning("Landmark model output is empty.")
            return None

        # Assuming the first output tensor contains the landmarks
        landmarks_output_tensor = outputs[0]
        self.logger.debug(f"Raw landmark_output_tensor shape: {landmarks_output_tensor.shape}")
        if landmarks_output_tensor.ndim < 2 or landmarks_output_tensor.shape[0] == 0:
             self.logger.error(f"Unexpected raw landmark_output_tensor shape: {landmarks_output_tensor.shape}")
             return None
        # Log first 5 rows of raw landmarks if possible
        log_max_rows = min(5, landmarks_output_tensor.shape[1] if landmarks_output_tensor.ndim > 1 else 1)
        self.logger.debug(f"Raw landmark_output_tensor (first {log_max_rows} landmarks, all channels from batch 0):\\n"
                          f"{landmarks_output_tensor[0, :log_max_rows, :] if landmarks_output_tensor.ndim > 2 else landmarks_output_tensor[0, :log_max_rows] if landmarks_output_tensor.ndim == 2 else landmarks_output_tensor[0]}")
        
        # Remove batch dimension
        if landmarks_output_tensor.ndim == 3 and landmarks_output_tensor.shape[0] == 1: # (1, num_landmarks, channels)
            landmarks_model_coords_raw = landmarks_output_tensor[0]
        elif landmarks_output_tensor.ndim == 2 and landmarks_output_tensor.shape[0] == 1: # (1, num_landmarks * channels) - needs reshape
            # This case needs careful handling if channels > 2
            self.logger.warning(f"Ambiguous landmark output shape (1, N*C): {landmarks_output_tensor.shape}. Assuming C=2 or C=3 and reshaping.")
            # Placeholder, assuming reshape to (N, C) is somehow known or C=2
            if landmarks_output_tensor.shape[1] % 2 == 0 and landmarks_output_tensor.shape[1] % 3 != 0 : # Prefer 2 channels if divisible
                 landmarks_model_coords_raw = landmarks_output_tensor[0].reshape(-1, 2)
            elif landmarks_output_tensor.shape[1] % 3 == 0:
                 landmarks_model_coords_raw = landmarks_output_tensor[0].reshape(-1, 3)
            else:
                 self.logger.error(f"Cannot reliably reshape (1, {landmarks_output_tensor.shape[1]}) into (N,2) or (N,3).")
                 return None
        elif landmarks_output_tensor.ndim == 2 : # (num_landmarks, channels) - less common for ONNX batch output but possible
            landmarks_model_coords_raw = landmarks_output_tensor
        else:
            self.logger.error(f"Unexpected landmark model output tensor shape after batch consideration: {landmarks_output_tensor.shape}. Expected (1, N, C) or (N,C).")
            return None
        
        self.logger.debug(f"landmarks_model_coords_raw (after batch removal, shape {landmarks_model_coords_raw.shape}, first {log_max_rows} landmarks):\\n"
                          f"{landmarks_model_coords_raw[:log_max_rows, :]}")

        num_landmarks = landmarks_model_coords_raw.shape[0]
        
        if landmarks_model_coords_raw.shape[1] < 2:
            self.logger.error(f"Landmark coordinates too few channels: {landmarks_model_coords_raw.shape}. Expected at least 2.")
            return None
        
        # Take only x, y coordinates (first 2 channels)
        landmarks_model_coords_xy = landmarks_model_coords_raw[:, :2]
        if landmarks_model_coords_raw.shape[1] > 2:
            self.logger.debug(f"Trimming extra landmark channels. Original channels: {landmarks_model_coords_raw.shape[1]}, using first 2.")
            
        self.logger.debug(f"landmarks_model_coords_xy (shape {landmarks_model_coords_xy.shape}, first {log_max_rows} landmarks):\\n"
                          f"{landmarks_model_coords_xy[:log_max_rows, :]}")

        # Normalize and scale coords from feature map to model input size
        target_h_model, target_w_model = self._input_shape
        assumed_feature_map_h = 64.0
        assumed_feature_map_w = 64.0
        coords_norm = landmarks_model_coords_xy / np.array([assumed_feature_map_w, assumed_feature_map_h])
        coords_scaled = coords_norm * np.array([target_w_model, target_h_model])
        # Map back to original image using inverse affine transform
        coords_scaled_batch = coords_scaled[np.newaxis, :, :].astype(np.float32)
        landmarks_full = cv2.transform(coords_scaled_batch, M_inv)[0]
        landmarks_full_image = landmarks_full.astype(np.int32)
        self.logger.info(f"Postprocessed {num_landmarks} landmarks. First {log_max_rows} final (x,y): {landmarks_full_image[:log_max_rows, :]}")
        return landmarks_full_image

    def extract_landmarks(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extracts facial landmarks for a given face bounding box in an image.

        Args:
            image (np.ndarray): The full image as a NumPy array (BGR).
            face_bbox (Tuple[int, int, int, int]): The bounding box (x, y, w, h)
                                                   of the face in the full image.

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (num_landmarks, 2)
                                  containing (x, y) coordinates for each landmark
                                  relative to the full original image. Returns None
                                  if landmarks cannot be extracted.
        """
        if self.landmark_model is None or self._input_name is None or self._output_names is None:
            self.logger.error("Landmark model not loaded or not properly configured. Cannot extract landmarks.")
            return None
        
        self.logger.debug(f"Extracting landmarks for face_bbox: {face_bbox} in image_shape: {image.shape}")

        try:
            preprocess_result = self._preprocess_face_roi(image, face_bbox)
            if preprocess_result is None:
                return None
            
            batch_face_roi, M_inv = preprocess_result
            self.logger.debug(f"Face ROI preprocessed. Input tensor shape: {batch_face_roi.shape}")

            outputs = self.landmark_model.run(self._output_names, {self._input_name: batch_face_roi})
            self.logger.debug(f"Landmark inference completed. Output shapes: {[o.shape for o in outputs]}")
            
            landmarks = self._postprocess_landmarks(outputs, M_inv)

            if landmarks is not None:
                self.logger.info(f"Successfully extracted {landmarks.shape[0]} landmarks.")
            else:
                self.logger.warning("Failed to postprocess landmarks.")
            return landmarks

        except Exception as e:
            self.logger.error(f"Error during landmark extraction for bbox {face_bbox}: {e}", exc_info=True)
            return None

# Exemplo de uso (seria chamado pelo Orchestrator ou em testes)
# if __name__ == '__main__':
#     from visiovox.core.logger_setup import setup_logging
#     setup_logging()
#     test_logger = logging.getLogger("landmark_extractor_test")

#     # Mock ConfigManager and ResourceManager
#     class MockCfgMgr:
#         def get(self, key, default=None):
#             test_logger.debug(f"MockConfigManager attempting to get: {key}")
#             if key == "model_catalog.face_landmarker.2dfan4.path_local_cache":
#                 return "models/face_landmarkers/2dfan4.onnx" # Ensure this file exists
#             if key == "model_catalog.face_landmarker.2dfan4.input_size_hw":
#                 return [256, 256] # Example for 2dfan4
#             # Add other model specific configs if needed by your _get_model_config in load_landmark_model
#             return default

#     class MockResMgr:
#         def __init__(self, cfg): self.cfg = cfg
#         def load_model(self, model_name, model_type):
#             return self.cfg.get(f"model_catalog.{model_type}.{model_name}.path_local_cache")

#     try:
#         test_logger.info("--- LandmarkExtractor Test ---")
#         cfg = MockCfgMgr()
#         res = MockResMgr(cfg)
#         extractor = LandmarkExtractor(config_manager=cfg, resource_manager=res)

#         # ** IMPORTANTE: Para este teste funcionar, você precisa:
#         # 1. Ter o modelo 'models/face_landmarkers/2dfan4.onnx' (ou o modelo padrão)
#         #    disponível e um ONNX válido. O download automático não é testado aqui.
#         # 2. Ter 'onnxruntime' e 'opencv-python' instalados.

#         if not os.path.exists(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")), "models/face_landmarkers/2dfan4.onnx")):
#             test_logger.error("O modelo 2dfan4.onnx não foi encontrado. Crie um placeholder ou coloque o modelo real.")
#         elif extractor.load_model(model_name="2dfan4"): # Ou outro modelo que você tenha
#             test_logger.info("Modelo de landmark carregado.")
            
#             # Criar uma imagem de teste e uma bbox de rosto simulada
#             # (idealmente, use uma imagem real e uma bbox de um FaceDetector real)
#             # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
#             # image_path = os.path.join(project_root, "data", "input", "sample_test_image.png") # Use sua imagem
#             # if os.path.exists(image_path):
#             #     test_image = cv2.imread(image_path)
#             # else:
#             #     test_logger.warning(f"Imagem de teste {image_path} não encontrada, usando dummy.")
#             #     test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
#             test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) # Dummy image
#             face_bbox_example = (100, 120, 200, 220) # x, y, w, h

#             # Desenhar a bbox na imagem dummy para visualização (opcional)
#             # cv2.rectangle(test_image, (face_bbox_example[0], face_bbox_example[1]), 
#             #               (face_bbox_example[0]+face_bbox_example[2], face_bbox_example[1]+face_bbox_example[3]), (0,255,0), 2)

#             landmarks = extractor.extract_landmarks(test_image, face_bbox_example)
#             if landmarks is not None:
#                 test_logger.info(f"Landmarks extraídos ({landmarks.shape[0]} pontos): {landmarks[:5]}...") # Mostra os 5 primeiros
#                 # Opcional: desenhar landmarks na imagem
#                 # for (lx, ly) in landmarks:
#                 #     cv2.circle(test_image, (lx, ly), 2, (0, 0, 255), -1)
#                 # cv2.imwrite("data/output/landmarks_test_output.jpg", test_image)
#                 # test_logger.info("Imagem com landmarks salva em data/output/landmarks_test_output.jpg")
#             else:
#                 test_logger.error("Falha ao extrair landmarks.")
#         else:
#             test_logger.error("Falha ao carregar o modelo de landmark.")

#     except ImportError as e:
#         test_logger.error(f"ImportError: {e}. Verifique as dependências (onnxruntime, opencv-python).")
#     except ConfigError as e:
#         test_logger.error(f"ConfigError: {e}")
#     except Exception as e:
#         test_logger.error(f"Erro inesperado no teste do LandmarkExtractor: {e}", exc_info=True)