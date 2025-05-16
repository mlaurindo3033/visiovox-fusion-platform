import logging
import os
from typing import Optional, List, Tuple

# Importações dos módulos do core
from .config_manager import ConfigManager, ConfigError
from .resource_manager import ResourceManager

# Importações dos módulos que serão usados (serão criados nos próximos passos)
# Usaremos caminhos absolutos do pacote 'visiovox'
from visiovox.modules.media_ingestion.media_loader import MediaLoader
from visiovox.modules.scene_analysis.face_detector import FaceDetector
from visiovox.modules.scene_analysis.landmark_extractor import LandmarkExtractor
from visiovox.modules.facial_processing.face_enhancer import FaceEnhancer
from visiovox.modules.scene_analysis.face_recogniser import FaceRecogniser
from visiovox.utils.image_utils import draw_detections_and_landmarks
from visiovox.modules.scene_analysis.mediapipe_face_mesh import MediaPipeFaceMeshExtractor

# Importações de bibliotecas de terceiros para funcionalidade específica (RF7.2)
import cv2 # OpenCV para desenhar bounding boxes
import numpy as np


class Orchestrator:
    """
    Orchestrates the main processing pipelines of the VisioVox application.
    It coordinates the different modules like media ingestion, scene analysis,
    facial processing, etc.
    """
    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager):
        """
        Initializes the Orchestrator.

        Args:
            config_manager (ConfigManager): Instance of the configuration manager.
            resource_manager (ResourceManager): Instance of the resource manager.
        """
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Instanciar módulos que são reutilizáveis ou têm estado gerenciado pelo Orchestrator
        try:
            self.media_loader = MediaLoader(config_manager=self.config_manager)
            self.face_detector = FaceDetector(config_manager=self.config_manager, resource_manager=self.resource_manager)
            self.landmark_extractor = LandmarkExtractor(config_manager=self.config_manager, resource_manager=self.resource_manager)
            self.face_enhancer = FaceEnhancer(config_manager=self.config_manager, resource_manager=self.resource_manager)
            self.face_recogniser = FaceRecogniser(config_manager=self.config_manager, resource_manager=self.resource_manager)
            self.logger.info("Initializing MediaPipeFaceMeshExtractor...")
            self.mp_landmarker = MediaPipeFaceMeshExtractor() # Primary landmarker
        except Exception as e:
            self.logger.critical(f"Failed to initialize core components in Orchestrator: {e}", exc_info=True)
            self.media_loader = None
            self.face_detector = None
            self.landmark_extractor = None
            self.face_enhancer = None
            self.face_recogniser = None

        self.logger.info("Orchestrator initialized.")

    def process_static_image(self, image_path: str, output_dir: str = "data/output") -> None:
        if not all([self.media_loader, self.face_detector, self.landmark_extractor, self.face_enhancer, self.face_recogniser]):
            self.logger.error("Orchestrator components not fully initialized. Aborting image processing.")
            return

        self.logger.info(f"Starting static image processing for: {image_path}")

        img_original_bgr = self.media_loader.load_image(image_path)
        if img_original_bgr is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return

        img_vis = img_original_bgr.copy()
        self.logger.info(f"Image loaded. Shape: {img_vis.shape}")

        # 1. Detectar faces
        default_detector_model = self.config_manager.get("default_models.face_detector", "yolo_default_onnx")
        if not self.face_detector.load_detection_model(default_detector_model):
            self.logger.error(f"Failed to load face detection model: {default_detector_model}. Aborting.")
            return
        
        detected_faces_bboxes_xywh = self.face_detector.detect_faces(img_vis)
        if not detected_faces_bboxes_xywh:
            self.logger.info("No faces detected.")
            self._save_image(img_vis, image_path, output_dir, "no_faces_detected")
            return
        self.logger.info(f"Detected {len(detected_faces_bboxes_xywh)} faces.")

        pipeline_face_data = []
        for x, y, w, h in detected_faces_bboxes_xywh:
            pipeline_face_data.append({
                "bbox_xywh": (x, y, w, h),
                "bbox_xyxy": (x, y, x + w, y + h),
                "cropped_face": None, "enhanced_face": None, 
                "landmarks": None, "embedding": None
            })

        # 2. Aprimorar faces
        default_enhancer_model = self.config_manager.get("default_models.face_enhancer", "gfpgan_1_4")
        enhancer_loaded = False
        if self.face_enhancer and default_enhancer_model:
            self.logger.info(f"Loading default face enhancer model: {default_enhancer_model}")
            enhancer_loaded = self.face_enhancer.load_model(enhancer_name=default_enhancer_model)
            if not enhancer_loaded:
                self.logger.warning(f"Failed to load default face enhancer '{default_enhancer_model}'. Enhancement will be skipped.")

        for i, face_data in enumerate(pipeline_face_data):
            x1, y1, x2, y2 = face_data["bbox_xyxy"]
            cropped_original = img_original_bgr[y1:y2, x1:x2]
            face_data["cropped_face"] = cropped_original

            if cropped_original.size == 0:
                self.logger.warning(f"Face {i} bbox {face_data['bbox_xyxy']} empty crop. Skipping.")
                face_data["enhanced_face"] = cropped_original # Fallback
                continue

            current_face_to_process = cropped_original
            if enhancer_loaded:
                enhanced_img = self.face_enhancer.enhance_face(img_original_bgr, face_data["bbox_xywh"])
                if enhanced_img is not None:
                    self.logger.debug(f"Face {i} enhanced.")
                    current_face_to_process = enhanced_img
                    # Update img_vis with the enhanced face
                    try:
                        img_vis[y1:y2, x1:x2] = cv2.resize(enhanced_img, (x2 - x1, y2 - y1))
                    except Exception as e_resize:
                        self.logger.error(f"Error resizing/pasting enhanced face {i}: {e_resize}")
                        current_face_to_process = cropped_original # Revert for safety
                    face_data["image_enhanced"] = enhanced_img # Storing for potential later use/saving
                    self.logger.info(f"Face {i} enhanced and updated in img_vis.")
                else:
                    self.logger.warning(f"Face enhancement failed for face {i} or returned None.")
            face_data["enhanced_face"] = current_face_to_process # This is what will be used for landmarking if enhanced, else original
        
            # --- Landmark Extraction (MediaPipe primary, ONNX fallback) ---
            if self.config_manager.get("pipeline_settings.run_landmark_extraction", True) and face_data.get("bbox_xyxy") is not None:
                landmarks = None
                bbox_xywh = face_data.get("bbox_xywh")
                bbox_xyxy = face_data.get("bbox_xyxy")

                # 1. Try MediaPipe
                if self.mp_landmarker and bbox_xywh:
                    self.logger.info(f"Attempting landmark extraction for face {i} using MediaPipe...")
                    try:
                        landmarks_mp = self.mp_landmarker.extract_landmarks(img_vis, bbox_xywh)
                        if landmarks_mp is not None and len(landmarks_mp) > 0:
                            self.logger.info(f"MediaPipe successfully extracted {len(landmarks_mp)} landmarks for face {i}. Sample: {landmarks_mp[:2]}.")
                            landmarks = landmarks_mp
                        else:
                            self.logger.warning(f"MediaPipe landmark extraction returned no points for face {i}.")
                    except Exception as e_mp:
                        self.logger.error(f"Exception during MediaPipe landmark extraction for face {i}: {e_mp}")

                # 2. Fallback to ONNX Landmarker
                landmarker_model_name = self.config_manager.get("models.face_landmarker.default_model_name", "2dfan4_default_onnx")
                if landmarks is None and self.landmark_extractor and self.landmark_extractor.is_model_loaded() and bbox_xyxy:
                    self.logger.warning(f"MediaPipe failed. Attempting fallback landmark extraction for face {i} using ONNX model '{landmarker_model_name}'.")
                    try:
                        landmarks_onnx = self.landmark_extractor.extract_landmarks(img_vis, bbox_xyxy)
                        if landmarks_onnx is not None and len(landmarks_onnx) > 0:
                            self.logger.info(f"ONNX LandmarkExtractor successfully extracted {len(landmarks_onnx)} landmarks for face {i}. Sample: {landmarks_onnx[:2]}.")
                            landmarks = landmarks_onnx
                        else:
                            self.logger.warning(f"ONNX landmark extraction also returned no points for face {i}.")
                    except Exception as e_onnx:
                        self.logger.error(f"Exception during ONNX landmark extraction for face {i}: {e_onnx}")
                
                if landmarks is not None:
                    self.logger.info(f"Successfully extracted {len(landmarks)} landmarks for face {i}.")
                    face_data["landmarks"] = landmarks
                else:
                    self.logger.error(f"All landmark extraction methods failed for face {i}.")
                    face_data["landmarks"] = None
            else:
                self.logger.info(f"Skipping landmark extraction for face {i} (config/no bbox).")
                face_data["landmarks"] = None
        
        # 4. Extrair embeddings
        default_recogniser_model = self.config_manager.get("default_models.face_recogniser", "arcface_w600k_r50")
        recogniser_loaded = False
        if self.face_recogniser and default_recogniser_model:
            self.logger.info(f"Loading default face recogniser model: {default_recogniser_model}")
            recogniser_loaded = self.face_recogniser.load_model(model_name=default_recogniser_model)
            if not recogniser_loaded:
                self.logger.warning(f"Failed to load default face recogniser '{default_recogniser_model}'. Recognition will be skipped.")

        if recogniser_loaded and all(face_data["landmarks"] is not None for face_data in pipeline_face_data): # Need landmarks for recognition
            for i, face_data in enumerate(pipeline_face_data):
                if face_data["landmarks"] is None:
                    self.logger.warning(f"Skipping embedding for face {i}, no landmarks.")
                    continue
                # FaceRecogniser.get_face_embedding expects the full image (img_vis) and landmarks for alignment
                embedding = self.face_recogniser.get_face_embedding(img_vis, face_data["landmarks"])
                if embedding is not None:
                    self.logger.info(f"Embedding for face {i}. Shape: {embedding.shape}. Preview: {embedding[:5]}")
                    face_data["embedding"] = embedding
                else:
                    self.logger.warning(f"Embedding extraction failed for face {i}.")

        # 5. Desenhar detecções e landmarks
        bboxes_to_draw_xywh = [fd["bbox_xywh"] for fd in pipeline_face_data if fd["enhanced_face"].size > 0]
        landmarks_to_draw = [fd["landmarks"] for fd in pipeline_face_data if fd["enhanced_face"].size > 0]
        
        if bboxes_to_draw_xywh:
            draw_detections_and_landmarks(img_vis, bboxes_to_draw_xywh, landmarks_to_draw)
            self.logger.info("Detections and landmarks drawn.")
        
        self._save_image(img_vis, image_path, output_dir, "processed")
        self.logger.info(f"Static image processing finished for: {image_path}")

    def _save_image(self, image_array: np.ndarray, original_path: str, output_dir: str, suffix: str):
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(original_path)
            name, ext = os.path.splitext(base_name)
            output_filename = os.path.join(output_dir, f"{name}_{suffix}{ext}")
            cv2.imwrite(output_filename, image_array)
            self.logger.info(f"Image saved to: {output_filename}")
        except Exception as e:
            self.logger.error(f"Error saving image to {output_dir}: {e}")

    def _draw_bounding_boxes_and_save(
        self,
        image_array: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        output_path: str
    ) -> None:
        """
        Helper method to draw bounding boxes on an image and save it.
        """
        try:
            for (x, y, w, h) in faces:
                cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")

            if cv2.imwrite(output_path, image_array):
                self.logger.info(f"Image with detected faces saved to: {output_path}")
            else:
                self.logger.error(f"Failed to save image to: {output_path}")

        except Exception as e:
            self.logger.error(f"Exception in _draw_bounding_boxes_and_save: {e}", exc_info=True)


# Exemplo de uso (seria em um main.py ou similar)
# if __name__ == '__main__':
#     from visiovox.core.logger_setup import setup_logging
#     setup_logging() # Configura o logging conforme logging_config.yaml

#     logger = logging.getLogger("visiovox.main_test") # Logger específico para este teste

#     # Supõe que configs/default_config.yaml está configurado corretamente
#     # e que os modelos e caminhos de saída existem ou podem ser criados.
#     # Para um teste real, você precisaria de um modelo ONNX em models/face_detection/
#     # e uma imagem de exemplo.

#     try:
#         logger.info("Initializing application components for Orchestrator test...")
#         cfg_manager = ConfigManager() # Usa configs/default_config.yaml
#         res_manager = ResourceManager(config_manager=cfg_manager)
#         orchestrator = Orchestrator(config_manager=cfg_manager, resource_manager=res_manager)
#         logger.info("Application components initialized.")

#         # Crie um arquivo de imagem de exemplo em data/input/sample_image.png ou .jpg
#         # por exemplo, 'data/input/sample_image.jpg'
#         sample_image_path_relative = "data/input/sample_image.jpg" # Coloque uma imagem aqui!
#         project_r = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#         sample_image_path_absolute = os.path.join(project_r, sample_image_path_relative)
        
#         if not os.path.exists(sample_image_path_absolute):
#             logger.error(f"SAMPLE IMAGE FOR TESTING NOT FOUND at {sample_image_path_absolute}")
#             logger.error("Please create a sample image (e.g., data/input/sample_image.jpg) to run this test.")
#         else:
#             logger.info(f"Attempting to process image: {sample_image_path_absolute}")
#             success = orchestrator.process_static_image(sample_image_path_absolute)
#             logger.info(f"Orchestrator.process_static_image result: {success}")

#     except ConfigError as e:
#         logger.critical(f"Configuration error during test setup: {e}", exc_info=True)
#     except ImportError as e:
#         logger.critical(f"ImportError: {e}. Make sure all modules (MediaLoader, FaceDetector) are created.", exc_info=True)
#     except Exception as e:
#         logger.critical(f"An unexpected error occurred during Orchestrator test: {e}", exc_info=True)
