import logging
import os
from typing import List, Tuple, Optional, Any, Dict

import cv2
import numpy as np
import onnxruntime

from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager


class FaceDetector:
    """
    Detects faces in an image using an ONNX model (e.g., YOLO).
    """

    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager):
        """
        Initializes the FaceDetector.

        Args:
            config_manager (ConfigManager): Instance of the configuration manager.
            resource_manager (ResourceManager): Instance of the resource manager.
        """
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.detection_model: Optional[onnxruntime.InferenceSession] = None
        self.model_name_loaded: Optional[str] = None # Para buscar configs específicas do modelo carregado
        
        # Informações do modelo que serão carregadas de config
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._input_shape: Optional[Tuple[int, int]] = None # (height, width)
        self._confidence_threshold: float = 0.5
        self._iou_threshold: float = 0.45
        
        self.logger.info("FaceDetector initialized.")

    def _get_model_config(self, model_name: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Helper to get model-specific config."""
        config_key = f"models.face_detection.{model_name}.{key}"
        value = self.config_manager.get(config_key, default)
        if value is None and default is None: # Se a chave não existir e não houver default
             self.logger.warning(f"Configuration key '{config_key}' not found for model '{model_name}'.")
        return value

    def load_detection_model(self, model_name: str = "yolo_default_onnx") -> bool:
        """
        Loads the specified ONNX face detection model using the ResourceManager
        and prepares it for inference.

        Args:
            model_name (str): The name of the model to load, as defined in the configuration
                              (e.g., "yolo_default_onnx").

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if self.detection_model and self.model_name_loaded == model_name:
            self.logger.info(f"Detection model '{model_name}' is already loaded.")
            return True
        
        self.logger.info(f"Attempting to load face detection model: {model_name}")
        
        model_data_from_res_mgr = self.resource_manager.load_model(model_name=model_name, model_type="face_detection")

        if not model_data_from_res_mgr:
            self.logger.error(f"Failed to get model path/info for detector '{model_name}' from ResourceManager.")
            return False

        model_details_to_use: Optional[Dict[str, Any]] = None
        path_relative_to_project_root: Optional[str] = None

        if isinstance(model_data_from_res_mgr, dict): # Model info loaded from manifest by ResourceManager
            model_details_to_use = model_data_from_res_mgr
            path_relative_to_project_root = model_details_to_use.get('path_local_cache')
            if not path_relative_to_project_root:
                self.logger.error(f"Manifest data for detector '{model_name}' is missing 'path_local_cache'.")
                return False
        elif isinstance(model_data_from_res_mgr, str): # Path string loaded from config or fallback by ResourceManager
            path_relative_to_project_root = model_data_from_res_mgr
            manifest_details = self.resource_manager.get_model_details_from_manifest(model_name, "face_detection")
            if manifest_details:
                self.logger.info(f"Found manifest details for '{model_name}' even though path was from config/fallback.")
                model_details_to_use = manifest_details
            else:
                self.logger.info(f"No separate manifest details found for '{model_name}'; path was from config/fallback.")
        else:
            self.logger.error(f"Invalid model data type '{type(model_data_from_res_mgr)}' received from ResourceManager for '{model_name}'.")
            return False
        
        try:
            project_root = self.config_manager.get_project_root()
            absolute_model_path = os.path.join(project_root, path_relative_to_project_root)

            if not os.path.exists(absolute_model_path):
                self.logger.error(f"ONNX detector model file not found at resolved path: {absolute_model_path}")
                # Consider adding a forced re-download attempt here if applicable
                # e.g., by calling self.resource_manager.load_model with force_download_check=True
                # For now, just failing. ResourceManager should ideally ensure file presence on successful load_model.
                return False

            self.logger.debug(f"Loading ONNX session from: {absolute_model_path}")
            
            providers = self.config_manager.get("onnx_providers", ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.logger.info(f"Using ONNX Runtime providers: {providers}")
            self.detection_model = onnxruntime.InferenceSession(absolute_model_path, providers=providers)
            self.model_name_loaded = model_name # Set this after successful model load

            # Configure model parameters using model_details_to_use or fallbacks
            if model_details_to_use:
                self.logger.info(f"Configuring detector '{model_name}' using details: {model_details_to_use}")
                self._input_name = model_details_to_use.get("input_name", self.detection_model.get_inputs()[0].name)
                
                output_names_val = model_details_to_use.get("output_names")
                if isinstance(output_names_val, list): self._output_names = output_names_val
                elif isinstance(output_names_val, str): self._output_names = [output_names_val]
                else: self._output_names = [o.name for o in self.detection_model.get_outputs()]

                input_shape_hw_val = model_details_to_use.get("input_size_hw")
                if isinstance(input_shape_hw_val, list) and len(input_shape_hw_val) == 2 and all(isinstance(dim, int) for dim in input_shape_hw_val):
                    self._input_shape = tuple(input_shape_hw_val) # type: ignore
                else: # Fallback for shape if not correctly in manifest
                    model_input_dims = self.detection_model.get_inputs()[0].shape
                    if len(model_input_dims) == 4 and all(isinstance(d, int) for d in model_input_dims[2:]): # type: ignore
                        self._input_shape = (model_input_dims[2], model_input_dims[3]) # H, W
                    else:
                        self._input_shape = tuple(self._get_model_config(model_name, "input_size_hw", default=[640, 640]))
                        self.logger.warning(f"Using configured/default input_shape {self._input_shape} for '{model_name}' (manifest/model inspection unclear).")

                self._confidence_threshold = float(model_details_to_use.get("confidence_threshold", self._get_model_config(model_name, "confidence_threshold", 0.5)))
                self._iou_threshold = float(model_details_to_use.get("iou_threshold", self._get_model_config(model_name, "iou_threshold", 0.45)))
            else: # Fallback if no model_details_to_use (e.g. model from config and no manifest entry)
                self.logger.warning(f"No detailed manifest/config for '{model_name}'. Using ONNX inspection and defaults.")
                self._input_name = self.detection_model.get_inputs()[0].name
                self._output_names = [o.name for o in self.detection_model.get_outputs()]
                
                model_input_dims = self.detection_model.get_inputs()[0].shape
                if len(model_input_dims) == 4 and all(isinstance(d, int) for d in model_input_dims[2:]): # type: ignore
                    self._input_shape = (model_input_dims[2], model_input_dims[3]) # H, W
                else: 
                    self._input_shape = tuple(self._get_model_config(model_name, "input_size_hw", default=[640, 640]))
                    self.logger.warning(f"Using configured/default input_shape {self._input_shape} for '{model_name}' (model inspection unclear).")
                
                self._confidence_threshold = float(self._get_model_config(model_name, "confidence_threshold", 0.5))
                self._iou_threshold = float(self._get_model_config(model_name, "iou_threshold", 0.45))

            self.logger.info(f"Face detection model '{model_name}' loaded. Input: '{self._input_name}', Outputs: {self._output_names}, Shape HW: {self._input_shape}, Conf: {self._confidence_threshold}, IoU: {self._iou_threshold}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model '{model_name}' from {absolute_model_path if 'absolute_model_path' in locals() else path_relative_to_project_root}: {e}", exc_info=True)
            self.detection_model = None
            self.model_name_loaded = None # Ensure this is cleared on failure
            return False

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocesses the image for YOLO ONNX model.
        Resizes, normalizes, transposes dimensions (HWC to CHW), and adds batch dimension.
        """
        if self._input_shape is None:
            raise ValueError("Model input shape not configured. Load model first.")
            
        target_h, target_w = self._input_shape
        original_h, original_w = image.shape[:2]

        # Redimensionar mantendo a proporção e adicionando padding se necessário,
        # ou simplesmente redimensionar (mais simples para começar).
        # Por simplicidade, vamos redimensionar diretamente.
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalizar para [0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Transpor de HWC para CHW (Height, Width, Channels -> Channels, Height, Width)
        chw_image = np.transpose(normalized_image, (2, 0, 1))
        
        # Adicionar dimensão de batch
        batch_image = np.expand_dims(chw_image, axis=0)
        
        scale_x = original_w / target_w
        scale_y = original_h / target_h
        
        # This might distort if aspect ratios are very different and letterboxing is not used.
        if resized_image.shape[0] != target_h or resized_image.shape[1] != target_w:
            # self.logger.debug(f"Forcing resize of letterboxed image from ({resized_image.shape[1]},{resized_image.shape[0]}) to ({target_w},{target_h})")
            padded_image = cv2.resize(resized_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            padded_image = resized_image # Should already be target_w, target_h if no padding was needed

        # HWC to CHW, BGR to RGB, Normalize to [0,1]
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1/255.0, swapRB=True)
        # Calculate scale factors mapping model input back to original image
        scale_x = original_w / target_w
        scale_y = original_h / target_h
        # self.logger.debug(f"Image preprocessed. Input tensor shape: {blob.shape}, scale_x: {scale_x}, scale_y: {scale_y}")
        return blob, scale_x, scale_y

    def _postprocess_detections(
        self, outputs: List[np.ndarray], scale_x: float, scale_y: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Postprocesses the raw output from the YOLO model to get bounding boxes.
        This is a generic implementation and might need adjustment for specific YOLO versions.
        Assumes output format where detections are [x_center, y_center, width, height, conf, class_probs...].
        Or for some models, outputs might be a list of [boxes, scores, class_ids].
        YOLOv8 typically has one output tensor of shape (batch_size, num_predictions, 4_coords + num_classes).
        Example assumes output[0] contains all predictions: (1, N, 4 + C) or (1, 4+C, N)
        """
        if not outputs or not isinstance(outputs[0], np.ndarray):
            self.logger.error("Invalid model output for postprocessing.")
            return []

        predictions_tensor = outputs[0]
        # self.logger.debug(f"Inference completed. Number of output tensors: {len(outputs)}. First output shape: {predictions_tensor.shape}")

        if predictions_tensor.shape != (1, 5, 8400) and predictions_tensor.shape != (1,8400,5): # (1, num_coords + num_classes + num_masks, num_dets)
            if predictions_tensor.ndim == 3 and predictions_tensor.shape[0] == 1 and predictions_tensor.shape[2] > predictions_tensor.shape[1]:
                 # Assume [1, features, detections] -> transpose to [1, detections, features]
                # self.logger.debug(f"Transposing predictions from {predictions_tensor.shape} to (1, {predictions_tensor.shape[2]}, {predictions_tensor.shape[1]}) for compatibility.")
                predictions_tensor = np.transpose(predictions_tensor, (0, 2, 1))
            else:
                self.logger.warning(f"Unexpected YOLO output shape: {predictions_tensor.shape}. Expected (1, 5, 8400) or (1, 8400, 5). Postprocessing might fail.")

        # Transpose if predictions are [1, 5, 8400]
        if predictions_tensor.shape[1] == 5 and predictions_tensor.shape[2] == 8400:
            # self.logger.debug(f"Transposing predictions from (5, 8400) to (8400, 5)")
            predictions = predictions_tensor[0].T # Shape becomes (8400, 5)
        elif predictions_tensor.shape[1] == 8400 and predictions_tensor.shape[2] == 5:
            predictions = predictions_tensor[0] # Shape is already (8400, 5)

        boxes = []
        confidences = []

        for i in range(predictions.shape[0]):
            cx, cy, w, h, conf = predictions[i]
            # self.logger.debug(f"Interpreting bbox coords ({cx:.2f}, {cy:.2f}, {w:.2f}, {h:.2f}) as absolute to model input_shape {self._input_shape}")

            if conf < self._confidence_threshold:
                continue

            # Heurística: se valores >1, interpretam pixels relativos; senão, normalizado
            if max(cx, cy, w, h) > 1.0:
                self.logger.debug(
                    f"Interpreting bbox coords ({cx:.2f}, {cy:.2f}, {w:.2f}, {h:.2f}) as absolute to model input_shape {self._input_shape}"
                )
                x0 = cx - w / 2
                y0 = cy - h / 2
                x_min_orig = int(x0 * scale_x)
                y_min_orig = int(y0 * scale_y)
                width_orig = int(w * scale_x)
                height_orig = int(h * scale_y)
            else:
                self.logger.debug(
                    f"Interpreting bbox coords ({cx:.2f}, {cy:.2f}, {w:.2f}, {h:.2f}) as normalized to model input_shape {self._input_shape}"
                )
                input_h_model, input_w_model = self._input_shape
                x0 = cx - w / 2
                y0 = cy - h / 2
                x_min_orig = int(x0 * input_w_model * scale_x)
                y_min_orig = int(y0 * input_h_model * scale_y)
                width_orig = int(w * input_w_model * scale_x)
                height_orig = int(h * input_h_model * scale_y)

            boxes.append([x_min_orig, y_min_orig, width_orig, height_orig])
            confidences.append(float(conf))

        if not boxes:
            return []

        # Aplicar Non-Maximum Suppression (NMS)
        # cv2.dnn.NMSBoxes requer [x_min, y_min, width, height]
        indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=self._confidence_threshold, nms_threshold=self._iou_threshold)
        
        final_boxes = []
        if len(indices) > 0:
            # 'indices' pode ser um array 2D [[idx1], [idx2]] ou 1D [idx1, idx2] dependendo da versão/contexto
            # Achatar para garantir que seja 1D
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            for i in indices:
                final_boxes.append(tuple(boxes[i])) # (x, y, w, h)
        
        return final_boxes

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in the given image array.

        Args:
            image (np.ndarray): The image (as a NumPy array in BGR format)
                                in which to detect faces.

        Returns:
            List[Tuple[int, int, int, int]]: A list of tuples, where each tuple
                                             represents a bounding box (x, y, w, h)
                                             for a detected face. Returns an empty
                                             list if no faces are detected or if
                                             the model is not loaded.
        """
        if self.detection_model is None or self._input_name is None or self._output_names is None or self._input_shape is None:
            self.logger.error("Detection model is not loaded or not properly configured. Cannot detect faces.")
            return []

        # self.logger.debug(f"Starting face detection on image with shape: {image.shape}") # Can be verbose
        original_shape_hw = image.shape[:2]
        
        # Preprocess image and get scale factors
        blob, scale_x, scale_y = self._preprocess(image)
        
        try:
            # self.logger.debug(f"Running inference with model: {self.model_name_loaded}")
            outputs = self.detection_model.run(self._output_names, {self._input_name: blob})
            # self.logger.debug(f"Raw model outputs received. Count: {len(outputs)}")

            # Post-process using correct scale_x, scale_y
            detected_boxes = self._postprocess_detections(outputs, scale_x, scale_y)
            
            # self.logger.debug(f"Detected {len(detected_boxes)} faces after post-processing.")
            return detected_boxes

        except Exception as e:
            self.logger.error(f"An error occurred during face detection: {e}", exc_info=True)
            return []

# Exemplo de uso (para teste local, seria chamado pelo Orchestrator)
# if __name__ == '__main__':
#     from visiovox.core.logger_setup import setup_logging
#     setup_logging() # Configura logging
    
#     # Mock ConfigManager e ResourceManager para teste
#     class MockConfigManager:
#         def get(self, key, default=None):
#             if key == "models.face_detection.yolo_test.path":
#                 # Crie um dummy ONNX model para teste ou use um real pequeno
#                 # Este path deve existir e ser um ONNX válido.
#                 # Exemplo: um modelo simples que sempre retorna algo.
#                 return "dummy_yolo.onnx" # PRECISA CRIAR ESTE ARQUIVO ONNX
#             if key == "models.face_detection.yolo_test.input_width": return 640
#             if key == "models.face_detection.yolo_test.input_height": return 640
#             if key == "models.face_detection.yolo_test.confidence_threshold": return 0.5
#             if key == "models.face_detection.yolo_test.iou_threshold": return 0.45
#             return default

#     class MockResourceManager:
#         def __init__(self, cfg_mgr): self.cfg_mgr = cfg_mgr
#         def load_model(self, model_name, model_type):
#             return self.cfg_mgr.get(f"models.{model_type}.{model_name}.path")

    # logger_main = logging.getLogger("main_test_fd")
    # try:
    #     # --- CUIDADO: Para rodar este teste, você precisa de um arquivo ONNX real ---
    #     # --- ou um mock muito bom de onnxruntime.InferenceSession.           ---
    #     # --- O código abaixo provavelmente falhará sem um 'dummy_yolo.onnx' válido ---
    #     # --- e a biblioteca onnxruntime instalada.                               ---

    #     logger_main.info("Criando dummy_yolo.onnx para teste (se não existir)...")
    #     # Tentar criar um ONNX mínimo, se possível, ou instruir o usuário.
    #     # Por simplicidade, vamos assumir que o usuário fornecerá ou pulará esta parte se falhar.
    #     dummy_onnx_path = "dummy_yolo.onnx"
    #     if not os.path.exists(dummy_onnx_path):
    #         logger_main.warning(f"'{dummy_onnx_path}' não encontrado. O teste pode falhar ou ser incompleto.")
    #         # Código para criar um ONNX simples aqui seria complexo demais para este exemplo.
    #         # Exemplo: onnx.save(onnx.helper.make_model(...), dummy_onnx_path)

    #     cfg = MockConfigManager()
    #     res = MockResourceManager(cfg)
    #     detector = FaceDetector(config_manager=cfg, resource_manager=res)

    #     if detector.load_detection_model(model_name="yolo_test"):
    #         logger_main.info("Modelo de detecção carregado para teste.")
    #         # Criar uma imagem de teste
    #         test_image = np.zeros((480, 640, 3), dtype=np.uint8) # Imagem preta
    #         cv2.rectangle(test_image, (100, 100), (200, 200), (0,255,0), -1) # Um "rosto" verde

    #         faces = detector.detect_faces(test_image)
    #         logger_main.info(f"Faces detectadas no teste: {faces}")
    #     else:
    #         logger_main.error("Falha ao carregar modelo de detecção para teste.")

    # except ImportError:
    #     logger_main.error("onnxruntime não está instalado. Pule o teste do FaceDetector ou instale a biblioteca.")
    # except Exception as e:
    #     logger_main.error(f"Erro no teste do FaceDetector: {e}", exc_info=True)