# src/visiovox/modules/scene_analysis/face_recogniser.py
import logging
import os
from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
import onnxruntime

from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager

class FaceRecogniser:
    """
    Recognises faces and extracts embeddings using an ONNX model (e.g., ArcFace).
    """

    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager):
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.recogniser_model: Optional[onnxruntime.InferenceSession] = None
        self.current_model_name: Optional[str] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.input_size_hw: Optional[Tuple[int, int]] = None # (H, W)
        
        self.logger.info("FaceRecogniser initialized.")

    def _get_model_config(self, model_name: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Helper to get model-specific config from the model_manifest."""
        config_key = f"model_catalog.face_recogniser.{model_name}.{key}" # Note: model_type is "face_recogniser"
        # Using direct attribute access for config_manager assuming it's a simple dict-like object or has a get method
        # For robustness, ensure ConfigManager has a .get(key, default) method similar to dictionaries
        # For now, assuming a simplified direct access or a placeholder for a .get method
        value = self.config_manager.get(config_key, default) # Corrected to use .get
        if value is None and default is None: # Check if value is None AND no default was provided
             self.logger.warning(f"Configuration key '{config_key}' not found for recognition model '{model_name}'.")
        return value

    def load_model(self, model_name: str) -> bool:
        """
        Loads the specified face recognition model.
        The model type is assumed to be 'face_recogniser'.
        """
        if self.recogniser_model and self.current_model_name == model_name:
            self.logger.info(f"Recognition model '{model_name}' is already loaded.")
            return True

        self.logger.info(f"Attempting to load face recognition model: {model_name}")
        model_path_or_info = self.resource_manager.load_model(model_name=model_name, model_type="face_recogniser")

        if not model_path_or_info:
            self.logger.error(f"Failed to get model path for recogniser '{model_name}' from ResourceManager.")
            return False

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
                self.logger.error(f"ONNX recogniser model file not found at resolved path: {absolute_model_path}")
                self.logger.info(f"Attempting to re-verify/download '{model_name}' via ResourceManager.")
                model_path_or_info_retry = self.resource_manager.load_model(
                    model_name=model_name, 
                    model_type="face_recogniser", 
                    force_download_check=True
                )
                if not model_path_or_info_retry or not os.path.exists(os.path.join(project_root, model_path_or_info_retry if isinstance(model_path_or_info_retry, str) else model_path_or_info_retry.get('path_local_cache', ''))):
                    self.logger.error(f"Failed to find or download recogniser model '{model_name}' after retry.")
                    return False
                if isinstance(model_path_or_info_retry, str):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry)
                elif isinstance(model_path_or_info_retry, dict):
                     absolute_model_path = os.path.join(project_root, model_path_or_info_retry['path_local_cache'])
            
            self.logger.debug(f"Loading ONNX recogniser session from: {absolute_model_path}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.recogniser_model = onnxruntime.InferenceSession(absolute_model_path, providers=providers)
            self.current_model_name = model_name
            self.logger.info(f"Recognition model '{model_name}' loaded successfully.")

            # Get model details from manifest (preferred) or config
            model_details = self.resource_manager.get_model_details_from_manifest(model_name, "face_recogniser")

            if model_details:
                self.input_size_hw = tuple(model_details.get('input_size_hw', [112, 112]))
                self.input_name = model_details.get('input_name', self.recogniser_model.get_inputs()[0].name)
                self.output_name = model_details.get('output_name', self.recogniser_model.get_outputs()[0].name)
                self.logger.info(f"Model '{model_name}' (from manifest): Input='{self.input_name}', Output='{self.output_name}', Expected Shape HW={self.input_size_hw}")
            else:
                # Fallback to direct model inspection and config if manifest details are missing
                self.input_name = self.recogniser_model.get_inputs()[0].name
                self.output_name = self.recogniser_model.get_outputs()[0].name
                # Try to get input_size_hw from config as a fallback (e.g. default_config.yaml)
                default_size = self.config_manager.get_config(f"models.face_recogniser.{model_name}.input_size_hw", default=[112,112])
                self.input_size_hw = tuple(default_size)
                self.logger.warning(f"Model '{model_name}' details not in manifest. Using inspection/config: Input='{self.input_name}', Output='{self.output_name}', Shape HW={self.input_size_hw}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX recogniser model '{model_name}' from {absolute_model_path if 'absolute_model_path' in locals() else model_path_manifest_relative}: {e}", exc_info=True)
            self.recogniser_model = None
            return False

    def _align_face(self, image_bgr: np.ndarray, landmarks_68_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Aligns a face using 5 specific landmarks derived from the 68 landmarks provided.
        Uses affine transformation to a standard size (e.g., 112x112 for ArcFace).
        
        Args:
            image_bgr (np.ndarray): The full BGR image.
            landmarks_68_xy (np.ndarray): Array of 68 landmark (x,y) coordinates.

        Returns:
            Optional[np.ndarray]: The aligned face image (BGR, HWC), or None if alignment fails.
        """
        if self.input_size_hw is None:
            self.logger.error("Input size for recogniser model not set. Cannot align face.")
            return None
        
        target_h, target_w = self.input_size_hw

        # Standard 5 points for ArcFace alignment based on typical 68-point landmark indices:
        # These are approximations. Actual indices might vary slightly based on landmark model.
        # Left eye: avg of 36-41 (e.g., center point 36 and 39, or just use one like 36)
        # Right eye: avg of 42-47 (e.g., center point 42 and 45, or just use one like 45)
        # Nose tip: 30
        # Left mouth corner: 48
        # Right mouth corner: 54
        
        # Simplified: pick specific points if available and plausible
        # Ensure landmarks_68_xy has enough points
        if landmarks_68_xy.shape[0] < 68:
            self.logger.error(f"Insufficient landmarks provided ({landmarks_68_xy.shape[0]}/68) for alignment.")
            return None

        # Using specific indices that are commonly robust for these features.
        # These might need adjustment depending on the exact 68-point model output.
        # Indices (0-based) for: left eye, right eye, nose tip, left mouth, right mouth
        # Commonly used for dlib-style 68 points often mapped for ArcFace alignment:
        # Left eye center: often around 36 or by averaging points like 37,38,40,41
        # Right eye center: often around 45 or by averaging points like 43,44,46,47
        # For simplicity, we use single points. Consider averaging for more robustness if needed.
        src_pts = np.array([
            landmarks_68_xy[36], # Left eye (outer corner or a representative point)
            landmarks_68_xy[45], # Right eye (outer corner or a representative point)
            landmarks_68_xy[30], # Nose tip
            landmarks_68_xy[48], # Left mouth corner
            landmarks_68_xy[54]  # Right mouth corner
        ], dtype=np.float32)

        # Standard destination points for a 112x112 aligned face (common for ArcFace)
        # These are normalized, then scaled to target_w, target_h if not 112x112
        # The values below are typical for 112x112 output directly.
        if target_w == 112 and target_h == 112:
            dst_pts = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]
            ], dtype=np.float32)
        else:
            # Scale destination points if target_w, target_h are different from 112x112
            # (This part is less common, usually alignment aims for a standard like 112x112)
            self.logger.warning(f"Target size {target_w}x{target_h} is not 112x112. Scaling ArcFace dst_pts.")
            scale_w = target_w / 112.0
            scale_h = target_h / 112.0
            dst_pts_112 = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]
            ], dtype=np.float32)
            dst_pts = dst_pts_112 * np.array([scale_w, scale_h], dtype=np.float32)

        try:
            # Estimate affine transform
            # Note: estimateAffinePartial2D is for similarity (translation, rotation, scale)
            # For full affine, estimateAffine2D. Similarity is usually preferred for face alignment.
            # Using estimateAffinePartial2D for more stability with fewer points.
            # OpenCV getAffineTransform needs 3 pairs of points for full affine.
            # For 5 points, estimateRigidTransform was used, but it's deprecated.
            # skimage.transform.SimilarityTransform().estimate(src_pts, dst_pts) is an alternative.
            # Using cv2.estimateAffinePartial2D which computes an optimal rigid transformation.
            
            # transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
            # Using skimage for potentially more robust similarity transform estimation with 5 points
            from skimage import transform as tf
            tform = tf.SimilarityTransform()
            if not tform.estimate(src_pts, dst_pts):
                self.logger.error("Failed to estimate similarity transform for face alignment.")
                return None
            transformation_matrix = tform.params[:2] # Get the 2x3 affine matrix part
            
            if transformation_matrix is None:
                self.logger.error("Failed to compute transformation matrix for face alignment (estimateAffinePartial2D returned None).")
                return None

            # Apply affine transformation
            aligned_face = cv2.warpAffine(image_bgr, transformation_matrix, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)
            self.logger.debug(f"Face aligned to {aligned_face.shape} using {len(src_pts)} landmarks.")
            return aligned_face
        except ImportError:
            self.logger.error("scikit-image is not installed. Required for SimilarityTransform in _align_face. Please install it: pip install scikit-image")
            return None
        except Exception as e:
            self.logger.error(f"Error during face alignment: {e}", exc_info=True)
            # Log the points if they might be an issue
            self.logger.debug(f"Src points for alignment: {src_pts}")
            self.logger.debug(f"Dst points for alignment: {dst_pts}")
            return None

    def _preprocess_aligned_face(self, aligned_face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocesses the aligned face for the recognition model (e.g., ArcFace).
        Converts BGR->RGB, normalizes to [-1, 1] or [0,1] based on model needs, and transposes to NCHW.
        ArcFace typically expects BGR, normalized to (img - 127.5) / 128.0.
        And input shape (1, 3, 112, 112).
        """
        if self.input_size_hw is None:
            self.logger.error("Input shape for recogniser model not set. Cannot preprocess.")
            return None

        try:
            # 1. Ensure correct size (should already be from alignment, but double check)
            # target_h, target_w = self.input_size_hw
            # if aligned_face_bgr.shape[0] != target_h or aligned_face_bgr.shape[1] != target_w:
            #     self.logger.warning(f"Aligned face shape {aligned_face_bgr.shape} differs from target {self.input_size_hw}. Resizing.")
            #     img_resized = cv2.resize(aligned_face_bgr, (target_w, target_h))
            # else:
            #     img_resized = aligned_face_bgr
            # Assuming alignment already produced the correct size.
            img_resized = aligned_face_bgr
            
            # 2. Convert to float32
            img_float32 = img_resized.astype(np.float32)
            
            # 3. Normalize for ArcFace: (pixel_value - 127.5) / 128.0
            # This maps [0, 255] to approximately [-1, 1]
            img_normalized = (img_float32 - 127.5) / 128.0
            
            # 4. Transpose HWC to CHW ( Channels, Height, Width )
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            
            # 5. Add batch dimension NCHW
            batch_input = np.expand_dims(img_chw, axis=0)
            
            self.logger.debug(f"Preprocessing for ArcFace successful. Output shape: {batch_input.shape}")
            return batch_input
        except Exception as e:
            self.logger.error(f"Error during aligned face preprocessing for recogniser: {e}", exc_info=True)
            return None

    def get_face_embedding(self, img_bgr, kps_5):
        # 1. Alinhar para 112x112 usando template ArcFace
        arcface_template = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        from skimage import transform as trans
        tform = trans.SimilarityTransform()
        tform.estimate(kps_5, arcface_template)
        M = tform.params[0:2]
        aligned = cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR)
        aligned = aligned.astype(np.float32)
        # 2. Normalizar para [-1,1]
        aligned = (aligned - 127.5) / 127.5
        # 3. Transpor para (1,3,112,112)
        aligned = np.transpose(aligned, (2, 0, 1))[None, ...]
        # Dump do input alinhado para debug/comparação
        try:
            np.save('data/output/debug_arcface_input.npy', aligned)
            self.logger.info(f"Input alinhado ArcFace salvo em data/output/debug_arcface_input.npy. Shape: {aligned.shape}, Dtype: {aligned.dtype}, Min: {aligned.min()}, Max: {aligned.max()}")
        except Exception as e:
            self.logger.warning(f"Falha ao salvar input alinhado ArcFace: {e}")
        # 4. Inferência ONNX
        embedding = self.recogniser_model.run([self.output_name], {self.input_name: aligned})[0]
        embedding = embedding.flatten()
        # 5. L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

# Test block - updated to use new load_model and path logic
if __name__ == '__main__':
    from visiovox.core.logger_setup import setup_logging
    setup_logging()
    test_logger_fr = logging.getLogger("face_recogniser_main_test")

    class MockConfigManagerFR:
        def __init__(self):
            self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            test_logger_fr.info(f"MockConfigManagerFR project root: {self._project_root}")
        
        def get_project_root(self):
            return self._project_root
        
        def get_config(self, key, default=None):
            test_logger_fr.debug(f"MockConfigManager-FR getting config: {key}")
            if key == "models.face_recogniser.arcface_w600k_r50.input_size_hw": return [112,112]
            if key == "models.face_recogniser.arcface_w600k_r50.input_name": return "input"
            if key == "models.face_recogniser.arcface_w600k_r50.output_name": return "output"
            return default
        
        def get(self, key, default=None): # For manifest access by ResourceManager in this test
            test_logger_fr.debug(f"MockConfigManager-FR getting manifest-like key: {key}")
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.path_local_cache":
                return "models/face_recognisers/arcface_w600k_r50.onnx"
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.sha256_hash_expected":
                return "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.download_url":
                return "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx"
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.input_size_hw":
                return [112,112]
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.input_name":
                return "input"
            if key == "model_catalog.face_recogniser.arcface_w600k_r50.output_name":
                return "output"
            if key == "model_catalog.face_recogniser": # To simulate the section exists
                 return { "arcface_w600k_r50": { 'description': 'Test ArcFace', 
                                               'path_local_cache': 'models/face_recognisers/arcface_w600k_r50.onnx', 
                                               'input_size_hw': [112,112], 'input_name':'input', 'output_name':'output'} }
            return default

    class MockResourceManagerFR:
        def __init__(self, cfg_mgr):
            self.config_manager = cfg_mgr
            self.logger = test_logger_fr
            self.model_download_manager = None # Not testing download here, assume file exists

        def load_model(self, model_name: str, model_type: str, force_download_check: bool = False):
            test_logger_fr.info(f"MockResourceManager-FR load_model called for {model_name}, type {model_type}")
            model_catalog_type_key = f"model_catalog.{model_type}"
            model_section = self.config_manager.get(model_catalog_type_key)

            if model_section and isinstance(model_section, dict) and model_name in model_section:
                model_entry = model_section[model_name]
                details = {
                    'path_local_cache': model_entry.get('path_local_cache'),
                    'input_size_hw': model_entry.get('input_size_hw'),
                    'input_name': model_entry.get('input_name'),
                    'output_name': model_entry.get('output_name')
                    # Outros campos do manifesto como sha256, download_url seriam usados pelo real ResourceManager
                }
                if not details['path_local_cache']:
                    test_logger_fr.error(f"path_local_cache not found in mock manifest for {model_type}.{model_name}")
                    return None

                full_model_path = os.path.join(self.config_manager.get_project_root(), details['path_local_cache'])
                os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
                if not os.path.exists(full_model_path):
                    test_logger_fr.warning(f"Mock model file {full_model_path} not found, creating dummy ONNX for test.")
                    try:
                        import onnx
                        from onnx import helper
                        from onnx import TensorProto
                        X = helper.make_tensor_value_info(details.get('input_name','input'), TensorProto.FLOAT, [None, 3] + details.get('input_size_hw',[112,112]))
                        Y = helper.make_tensor_value_info(details.get('output_name','output'), TensorProto.FLOAT, [None, 512])
                        node_def = helper.make_node('Identity', [details.get('input_name','input')], [details.get('output_name','output')])
                        graph_def = helper.make_graph([node_def], 'dummy-model', [X], [Y])
                        model_def = helper.make_model(graph_def, producer_name='onnx-example')
                        onnx.save(model_def, full_model_path)
                        test_logger_fr.info(f"Created dummy ONNX model at {full_model_path}")
                    except Exception as e_onnx_dummy:
                        test_logger_fr.error(f"Could not create dummy ONNX: {e_onnx_dummy}")
                        return None 
                return details 
            test_logger_fr.error(f"Model {model_name} of type {model_type} not found in mock manifest via {model_catalog_type_key}")
            return None
        
        def get_model_details_from_manifest(self, model_name: str, model_type: str):
             # No mock, load_model já retorna os detalhes necessários que viriam do manifesto.
            model_info = self.load_model(model_name, model_type)
            if isinstance(model_info, dict): # load_model no mock retorna o dict de detalhes
                return model_info
            return None # Ou uma estrutura esperada se load_model retornasse apenas o caminho

    try:
        test_logger_fr.info("--- FaceRecogniser Isolated Test ---")
        mock_cfg_fr = MockConfigManagerFR()
        # Atualizar mock_res_fr para passar mock_cfg_fr
        mock_res_fr = MockResourceManagerFR(cfg_mgr=mock_cfg_fr) 
        face_recogniser_instance = FaceRecogniser(config_manager=mock_cfg_fr, resource_manager=mock_res_fr)

        test_model_name = "arcface_w600k_r50"
        model_loaded = face_recogniser_instance.load_model(model_name=test_model_name)
        assert model_loaded, f"Model '{test_model_name}' FAILED to load."
        test_logger_fr.info(f"Model '{test_model_name}' loaded successfully for test.")

        dummy_image_full_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_landmarks_68xy = np.zeros((68, 2), dtype=np.float32)
        base_x, base_y = 200, 200 
        pts_to_set = {
            30: (base_x + 56, base_y + 40), 36: (base_x + 30, base_y + 50),
            45: (base_x + 82, base_y + 50), 48: (base_x + 40, base_y + 80),
            54: (base_x + 72, base_y + 80) 
        }
        for i in range(68):
            if i in pts_to_set: dummy_landmarks_68xy[i] = pts_to_set[i]
            else: dummy_landmarks_68xy[i] = (base_x + np.random.randint(20, 90), base_y + np.random.randint(30, 100))
        
        test_logger_fr.info(f"Using dummy image shape: {dummy_image_full_bgr.shape} and dummy landmarks shape: {dummy_landmarks_68xy.shape}")
        embedding = face_recogniser_instance.get_face_embedding(dummy_image_full_bgr, dummy_landmarks_68xy)

        if embedding is not None:
            test_logger_fr.info(f"Successfully extracted embedding. Shape: {embedding.shape}")
            assert embedding.shape == (512,), f"Embedding shape {embedding.shape} is not the expected (512,)"
        else:
            assert False, "Embedding extraction failed in test."
        
        test_logger_fr.info("--- FaceRecogniser Isolated Test COMPLETED SUCCESSFULLY ---")

    except AssertionError as ae:
        test_logger_fr.error(f"TEST ASSERTION FAILED: {ae}", exc_info=True)
    except ConfigError as ce:
        test_logger_fr.critical(f"Configuration error in FaceRecogniser test: {ce}", exc_info=True)
    except ImportError as ie:
        test_logger_fr.critical(f"ImportError: {ie}. Check scikit-image: pip install scikit-image", exc_info=True)
    except Exception as ex:
        test_logger_fr.critical(f"An unexpected error occurred: {ex}", exc_info=True) 