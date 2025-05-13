import logging
import os
from typing import Optional, List, Tuple

# Importações dos módulos do core
from .config_manager import ConfigManager, ConfigError
from .resource_manager import ResourceManager

# Importações dos módulos que serão usados (serão criados nos próximos passos)
# Usaremos caminhos absolutos do pacote 'visiovox'
from visiovox.modules.media_ingestion.loader import MediaLoader
from visiovox.modules.scene_analysis.face_detector import FaceDetector

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
        except Exception as e:
            self.logger.critical(f"Failed to initialize core components (MediaLoader/FaceDetector) in Orchestrator: {e}", exc_info=True)
            self.media_loader = None
            self.face_detector = None

        self.logger.info("Orchestrator initialized.")

    def process_static_image(self, image_path: str) -> bool:
        """
        Processes a static image: loads it, detects faces, and optionally saves the result.

        Args:
            image_path (str): The path to the static image to be processed.

        Returns:
            bool: True if the processing pipeline completes successfully (even if no faces are detected),
                  False if a critical error occurs during the process.
        """
        self.logger.info(f"Received request to process static image: {image_path}")

        if not (image_path and isinstance(image_path, str)):
            self.logger.error(f"Invalid image path provided: '{image_path}'. Must be a non-empty string.")
            return False

        if not self.media_loader:
            self.logger.error("MediaLoader not available in Orchestrator. Cannot process image.")
            return False
        if not self.face_detector:
            self.logger.error("FaceDetector not available in Orchestrator. Cannot process image.")
            return False

        # 1. Carregar a imagem (RF7.1)
        self.logger.debug(f"Attempting to load image from: {image_path}")
        image_array: Optional[np.ndarray] = self.media_loader.load_static_image(image_path)

        if image_array is None:
            self.logger.error(f"Failed to load image from path: {image_path}. Orchestration aborted.")
            return False
        self.logger.info(f"Image loaded successfully from: {image_path}, shape: {image_array.shape}")

        # 2. Carregar o modelo de detecção de face (RF7.1)
        default_detector_model_name = self.config_manager.get("default_models.face_detector", "yolo_default_onnx")
        self.logger.debug(f"Attempting to load face detection model: {default_detector_model_name}")
        
        if not self.face_detector.load_detection_model(model_name=default_detector_model_name):
            self.logger.error(f"Failed to load face detection model '{default_detector_model_name}'. Orchestration aborted.")
            return False
        self.logger.info(f"Face detection model '{default_detector_model_name}' loaded successfully.")

        # 3. Detectar faces (RF7.1)
        self.logger.debug("Attempting to detect faces in the loaded image.")
        detected_faces: List[Tuple[int, int, int, int]] = self.face_detector.detect_faces(image_array)

        if detected_faces:
            self.logger.info(f"Detected {len(detected_faces)} face(s): {detected_faces}")
        else:
            self.logger.info("No faces detected in the image.")

        # 4. (Opcional) Desenhar bounding boxes e salvar (RF7.2)
        try:
            output_image_path_key = "output_paths.detected_faces_image"
            output_image_path_relative = self.config_manager.get(output_image_path_key)

            if output_image_path_relative and detected_faces:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                absolute_output_path = os.path.join(project_root, output_image_path_relative)
                
                self.logger.info(f"Attempting to draw bounding boxes and save to: {absolute_output_path}")
                self._draw_bounding_boxes_and_save(image_array.copy(), detected_faces, absolute_output_path)
            elif not output_image_path_relative:
                self.logger.debug(f"Output path key '{output_image_path_key}' not found in config. Skipping save.")

        except ConfigError as e:
            self.logger.warning(f"Configuration error related to output path: {e}. Skipping save with bounding boxes.")
        except Exception as e:
            self.logger.error(f"Error during drawing/saving bounding boxes: {e}", exc_info=True)

        self.logger.info(f"Static image processing pipeline completed for: {image_path}")
        return True

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
