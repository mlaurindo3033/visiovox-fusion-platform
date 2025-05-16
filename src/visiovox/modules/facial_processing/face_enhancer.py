# src/visiovox/modules/facial_processing/face_enhancer.py
import logging
import os
from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
import onnxruntime

from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager

class FaceEnhancer:
    """
    Enhances the quality of a detected face region using an ONNX model (e.g., GFPGAN).
    """

    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager):
        """
        Initializes the FaceEnhancer.

        Args:
            config_manager (ConfigManager): Instance of the configuration manager.
            resource_manager (ResourceManager): Instance of the resource manager.
        """
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.enhancer_model: Optional[onnxruntime.InferenceSession] = None
        self.model_name_loaded: Optional[str] = None
        self._input_name: str = "input" # Default, common for GFPGAN
        self._output_names: List[str] = ["output"] # Default, common for GFPGAN
        self._input_shape_hw: Optional[Tuple[int, int]] = None # (height, width)
        
        self.logger.info("FaceEnhancer initialized.")

    def _get_model_config(self, model_name: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Helper to get model-specific config from the model_manifest via ConfigManager."""
        config_key = f"model_catalog.face_enhancers.{model_name}.{key}"
        value = self.config_manager.get(config_key, default)
        if value is None and default is None:
             self.logger.warning(f"Configuration key '{config_key}' not found for enhancer model '{model_name}'.")
        return value

    def load_model(self, enhancer_name: str) -> bool:
        """
        Loads the specified face enhancement model.
        The model type is assumed to be 'face_enhancer'.
        """
        if self.enhancer_model and self.model_name_loaded == enhancer_name:
            self.logger.info(f"Enhancer model '{enhancer_name}' is already loaded.")
            return True

        self.logger.info(f"Attempting to load face enhancement model: {enhancer_name}")
        model_path_or_info = self.resource_manager.load_model(model_name=enhancer_name, model_type="face_enhancer")
        
        if not model_path_or_info:
            self.logger.error(f"Failed to get model path for enhancer '{enhancer_name}' from ResourceManager.")
            return False

        if isinstance(model_path_or_info, str):
            model_path_manifest_relative = model_path_or_info
        elif isinstance(model_path_or_info, dict) and 'path_local_cache' in model_path_or_info:
            model_path_manifest_relative = model_path_or_info['path_local_cache']
        else:
            self.logger.error(f"Invalid model info received from ResourceManager for '{enhancer_name}'. Expected str or dict with 'path_local_cache'.")
            return False
            
        try:
            project_root = self.config_manager.get_project_root()
            absolute_model_path = os.path.join(project_root, model_path_manifest_relative)
            
            if not os.path.exists(absolute_model_path):
                self.logger.error(f"ONNX enhancer model file not found at resolved path: {absolute_model_path}")
                self.logger.info(f"Attempting to re-verify/download '{enhancer_name}' via ResourceManager.")
                model_path_or_info_retry = self.resource_manager.load_model(
                    model_name=enhancer_name, 
                    model_type="face_enhancer", 
                    force_download_check=True
                )
                if not model_path_or_info_retry or not os.path.exists(os.path.join(project_root, model_path_or_info_retry if isinstance(model_path_or_info_retry, str) else model_path_or_info_retry.get('path_local_cache', ''))):
                    self.logger.error(f"Failed to find or download enhancer model '{enhancer_name}' after retry.")
                    return False
                if isinstance(model_path_or_info_retry, str):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry)
                elif isinstance(model_path_or_info_retry, dict):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry['path_local_cache'])


            self.logger.debug(f"Loading ONNX enhancer session from: {absolute_model_path}")
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.enhancer_model = onnxruntime.InferenceSession(absolute_model_path, providers=providers)
            
            self.model_name_loaded = enhancer_name
            self.logger.info(f"Face enhancement model '{enhancer_name}' loaded successfully.")

            model_details = self.resource_manager.get_model_details_from_manifest(enhancer_name, "face_enhancer")
            if model_details and 'input_size_hw' in model_details:
                self._input_shape_hw = tuple(model_details['input_size_hw'])
                self.logger.info(f"Model '{enhancer_name}' input size set to: {self._input_shape_hw}")
            else:
                default_input_size = self.config_manager.get_config(f"models.face_enhancer.{enhancer_name}.input_size_hw", default=[512, 512])
                self._input_shape_hw = tuple(default_input_size)
                self.logger.warning(f"Input size for '{enhancer_name}' not found in manifest, using default/config: {self._input_shape_hw}")

            # Get model input/output details (can be overridden by manifest)
            self._input_name = self._get_model_config(enhancer_name, "input_name", self.enhancer_model.get_inputs()[0].name)
            self_output_names_config = self._get_model_config(enhancer_name, "output_names")
            if isinstance(self_output_names_config, list):
                self._output_names = self_output_names_config
            else:
                # self.logger.debug(f"Output names for '{enhancer_name}' not found or not a list in manifest/config. Falling back to model's default outputs.")
                self._output_names = [output.name for output in self.enhancer_model.get_outputs()]

            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX enhancer model '{enhancer_name}' from {absolute_model_path if 'absolute_model_path' in locals() else model_path_manifest_relative}: {e}", exc_info=True)
            self.enhancer_model = None
            return False

    def _preprocess_face(self, face_roi_bgr_hwc: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocesses the face ROI for the enhancement model (e.g., GFPGAN).
        Resizes, converts BGR->RGB, normalizes to [-1, 1], and transposes to NCHW.
        """
        if self._input_shape_hw is None:
            self.logger.error("Input shape for enhancer model not set. Load model first. Cannot preprocess.")
            return None
        
        try:
            target_h, target_w = self._input_shape_hw
            
            # 1. Resize to model's expected input size
            resized_roi = cv2.resize(face_roi_bgr_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. Convert BGR to RGB
            img_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
            
            # 3. Convert to float32
            img_float32 = img_rgb.astype(np.float32)
            
            # 4. Scale to [0, 1]
            img_scaled_01 = img_float32 / 255.0
            
            # 5. Normalize to [-1, 1]
            img_normalized_neg1_pos1 = (img_scaled_01 - 0.5) * 2.0
            
            # 6. Transpose HWC to CHW
            img_chw = np.transpose(img_normalized_neg1_pos1, (2, 0, 1))
            
            # 7. Add batch dimension NCHW
            batch_input = np.expand_dims(img_chw, axis=0)
            
            # self.logger.debug(f"Preprocessing successful. Output shape: {batch_input.shape}")
            return batch_input
        except Exception as e:
            self.logger.error(f"Error during face preprocessing for enhancer: {e}", exc_info=True)
            return None

    def _postprocess_output(self, model_output_nchw: np.ndarray) -> Optional[np.ndarray]:
        """
        Postprocesses the NCHW output from the enhancement model back to a BGR HWC uint8 image.
        Input is assumed to be in the range [-1, 1].
        """
        try:
            if model_output_nchw.ndim == 4 and model_output_nchw.shape[0] == 1:
                output_chw = model_output_nchw[0] # Remove batch dimension
            elif model_output_nchw.ndim == 3: # Already CHW
                output_chw = model_output_nchw
            else:
                self.logger.error(f"Unexpected model output shape for postprocessing: {model_output_nchw.shape}")
                return None
                
            # 1. Clip to [-1, 1] (good practice, though model should ideally output this range)
            output_chw_clipped = np.clip(output_chw, -1.0, 1.0)
            
            # 2. Denormalize from [-1, 1] to [0, 1]
            output_chw_denorm_01 = (output_chw_clipped + 1.0) / 2.0
            
            # 3. Transpose CHW to HWC
            output_hwc_rgb = np.transpose(output_chw_denorm_01, (1, 2, 0))
            
            # 4. Scale to [0, 255]
            output_hwc_rgb_0_255 = output_hwc_rgb * 255.0
            
            # 5. Clip to [0, 255] and convert to uint8
            output_hwc_rgb_uint8 = np.clip(output_hwc_rgb_0_255, 0, 255).astype(np.uint8)
            
            # 6. Convert RGB to BGR (for OpenCV compatibility)
            output_hwc_bgr = cv2.cvtColor(output_hwc_rgb_uint8, cv2.COLOR_RGB2BGR)
            
            # self.logger.debug(f"Postprocessing successful. Output image shape: {output_hwc_bgr.shape}")
            return output_hwc_bgr
        except Exception as e:
            self.logger.error(f"Error during enhancer output postprocessing: {e}", exc_info=True)
            return None

    def enhance_face(self, full_image_bgr_hwc: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Enhances a single face within an image.

        Args:
            full_image_bgr_hwc (np.ndarray): The full original image (BGR, HWC, uint8).
            face_bbox (Tuple[int, int, int, int]): Bounding box (x, y, w, h) of the face.

        Returns:
            Optional[np.ndarray]: The enhanced face ROI, resized to original bbox dimensions (BGR, HWC, uint8),
                                  or None if enhancement fails.
        """
        if self.enhancer_model is None or self._input_name is None or self._output_names is None or self._input_shape_hw is None:
            self.logger.error("Enhancer model not loaded or not properly configured. Cannot enhance face.")
            return None

        x, y, w, h = face_bbox
        if w <= 0 or h <= 0:
            self.logger.warning(f"Invalid face_bbox for enhancement (width or height is zero or negative): {face_bbox}")
            return None

        # 1. Crop the face ROI from the full image
        img_h_full, img_w_full = full_image_bgr_hwc.shape[:2]
        roi_x1, roi_y1 = max(0, x), max(0, y)
        roi_x2, roi_y2 = min(img_w_full, x + w), min(img_h_full, y + h)
        
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            self.logger.warning(f"Face bbox {face_bbox} is outside image bounds or results in empty ROI after clipping.")
            return None
            
        face_roi_original = full_image_bgr_hwc[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if face_roi_original.size == 0:
            self.logger.warning(f"Cropped face_roi for enhancement is empty. Original Bbox: {face_bbox}, Clipped ROI coords: ({roi_x1},{roi_y1},{roi_x2},{roi_y2})")
            return None
        
        original_roi_h, original_roi_w = face_roi_original.shape[:2]
        self.logger.debug(f"Enhancing face ROI of original size ({original_roi_w},{original_roi_h}) at ({roi_x1},{roi_y1})")

        # 2. Preprocess the cropped face
        preprocessed_face_batch = self._preprocess_face(face_roi_original)
        if preprocessed_face_batch is None:
            self.logger.error("Preprocessing of face ROI failed for enhancer.")
            return None

        # 3. Run inference
        try:
            self.logger.debug(f"Running enhancement model '{self.model_name_loaded}' on preprocessed face shape {preprocessed_face_batch.shape}")
            raw_outputs = self.enhancer_model.run(self._output_names, {self._input_name: preprocessed_face_batch})
            
            # Assuming the primary enhanced image is the first output
            enhanced_face_model_size_bgr = self._postprocess_output(raw_outputs[0]) 
            if enhanced_face_model_size_bgr is None:
                self.logger.error("Postprocessing of enhanced face failed.")
                return None
            # self.logger.debug(f"Enhanced face (model output size {enhanced_face_model_size_bgr.shape}) successfully postprocessed.")

            # 4. Resize back to the original ROI size
            # self.logger.debug(f"Resizing enhanced face from {enhanced_face_model_size_bgr.shape[:2]} to original ROI size ({original_roi_w},{original_roi_h})")
            resized_enhanced_face = cv2.resize(enhanced_face_model_size_bgr, (original_roi_w, original_roi_h), interpolation=cv2.INTER_AREA)
            
            self.logger.info(f"Face ROI successfully enhanced and resized to original ROI dimensions ({original_roi_w},{original_roi_h}).")
            return resized_enhanced_face
        except Exception as e:
            self.logger.error(f"Error during enhancement model inference or final resize: {e}", exc_info=True)
            return None

# Bloco de exemplo para teste isolado (descomente e adapte para testar)
# if __name__ == '__main__':
#     from visiovox.core.logger_setup import setup_logging
#     setup_logging()
#     test_logger_enhancer = logging.getLogger("face_enhancer_main_test")

#     # Mock ConfigManager e ResourceManager para este teste isolado
#     # Em uma aplicação real, eles seriam instanciados e passados corretamente.
#     class MockConfigManagerEnhancer:
#         def get(self, key, default=None):
#             test_logger_enhancer.debug(f"MockConfigManager-Enhancer getting: {key}")
#             if key == "model_catalog.face_enhancers.gfpgan_1_4.path_local_cache":
#                 # Certifique-se que este modelo ONNX exista no caminho para o teste
#                 return "models/face_enhancers/gfpgan_1_4.onnx" 
#             if key == "model_catalog.face_enhancers.gfpgan_1_4.input_size_hw":
#                 return [512, 512]
#             # Adicione outros mocks de config se o FaceEnhancer os usar (ex: input_name, output_names)
#             return default

#     class MockResourceManagerEnhancer:
#         def __init__(self, cfg): self.cfg = cfg
#         def load_model(self, model_name, model_type):
#             # Simula o ResourceManager que já baixou o modelo e retorna o caminho do cache local
#             return self.cfg.get(f"model_catalog.{model_type}.{model_name}.path_local_cache")

#     try:
#         test_logger_enhancer.info("--- FaceEnhancer Isolated Test ---")
#         mock_cfg = MockConfigManagerEnhancer()
#         mock_res = MockResourceManagerEnhancer(mock_cfg)
#         face_enhancer_instance = FaceEnhancer(config_manager=mock_cfg, resource_manager=mock_res)

#         # Garanta que o modelo gfpgan_1.4.onnx está em models/face_enhancers/
#         # Se o download automático não for testado aqui, o arquivo deve estar presente.
#         # Para testar o download, use o ResourceManager real e o ConfigManager real.
#         model_is_loaded = face_enhancer_instance.load_model(model_name="gfpgan_1_4")
        
#         if model_is_loaded:
#             test_logger_enhancer.info("GFPGAN v1.4 model loaded successfully for test.")
            
#             # Carregue uma imagem de teste real ou crie uma dummy
#             # Lembre-se que o Orchestrator faria isso usando MediaLoader
#             project_r_enh = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
#             test_image_path_enh = os.path.join(project_r_enh, "data", "input", "sample_test_image.png") # Use sua imagem real

#             if not os.path.exists(test_image_path_enh):
#                 test_logger_enhancer.error(f"Imagem de teste NÃO ENCONTRADA: {test_image_path_enh}. Crie uma para o teste.")
#                 test_full_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) # Imagem dummy maior
#                 face_bbox_test_enh = (300, 300, 200, 200) # bbox dentro da imagem dummy
#                 cv2.rectangle(test_full_image, 
#                               (face_bbox_test_enh[0], face_bbox_test_enh[1]), 
#                               (face_bbox_test_enh[0] + face_bbox_test_enh[2], face_bbox_test_enh[1] + face_bbox_test_enh[3]), 
#                               (0, 255, 0), 2) # Desenha uma bbox na imagem dummy
#             else:
#                 test_full_image = cv2.imread(test_image_path_enh)
#                 if test_full_image is None:
#                     test_logger_enhancer.error(f"Falha ao carregar imagem de teste: {test_image_path_enh}")
#                     exit()
#                 # Simule uma bbox detectada (idealmente, use o FaceDetector para obter uma bbox real)
#                 # Para a sample_test_image.png que usamos antes, a bbox era (97, 115, 644, 862)
#                 # Ajuste esta bbox para corresponder a um rosto na sua imagem de teste
#                 face_bbox_test_enh = (97, 115, 644, 862) # x, y, w, h from previous test

#             test_logger_enhancer.info(f"Processando face com bbox: {face_bbox_test_enh} na imagem com shape: {test_full_image.shape}")
#             enhanced_face_roi = face_enhancer_instance.enhance_face(test_full_image, face_bbox_test_enh)

#             if enhanced_face_roi is not None:
#                 test_logger_enhancer.info(f"Aprimoramento da ROI da face bem-sucedido! Shape da ROI aprimorada: {enhanced_face_roi.shape}")
                
#                 # Salvar a ROI aprimorada e/ou a imagem completa com a ROI colada de volta
#                 cv2.imwrite("data/output/isolated_enhanced_face_roi.jpg", enhanced_face_roi)
#                 test_logger_enhancer.info("ROI aprimorada salva em data/output/isolated_enhanced_face_roi.jpg")

#                 # Colar de volta na imagem original para visualização completa
#                 x, y, w, h = face_bbox_test_enh
#                 # Certifique-se que enhanced_face_roi tem as dimensões w, h corretas
#                 if enhanced_face_roi.shape[0] == h and enhanced_face_roi.shape[1] == w:
#                     full_image_with_enhancement = test_full_image.copy()
#                     full_image_with_enhancement[y:y+h, x:x+w] = enhanced_face_roi
#                     cv2.imwrite("data/output/full_image_with_enhancement.jpg", full_image_with_enhancement)
#                     test_logger_enhancer.info("Imagem completa com face aprimorada salva em data/output/full_image_with_enhancement.jpg")
#                 else:
#                     test_logger_enhancer.error(f"Shape da ROI aprimorada ({enhanced_face_roi.shape}) não corresponde à bbox original ({w},{h}). Falha ao colar.")
#             else:
#                 test_logger_enhancer.error("Falha no aprimoramento da face.")
#         else:
#             test_logger_enhancer.error("Falha ao carregar o modelo de aprimoramento para o teste.")

#     except ConfigError as ce:
#         test_logger_enhancer.critical(f"Erro de Configuração no teste: {ce}", exc_info=True)
#     except ImportError as ie:
#         test_logger_enhancer.critical(f"Erro de Importação: {ie}. Verifique as dependências.", exc_info=True)
#     except Exception as ex:
#         test_logger_enhancer.critical(f"Erro inesperado no teste do FaceEnhancer: {ex}", exc_info=True) 