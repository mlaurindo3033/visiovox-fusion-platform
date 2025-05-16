# src/visiovox/modules/facial_processing/face_swapper.py
import logging
import os
from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
import onnxruntime
import torch

from visiovox.core.config_manager import ConfigManager, ConfigError
from visiovox.core.resource_manager import ResourceManager
# Precisaremos de funções de alinhamento, que podem vir de utils ou serem definidas aqui.
# from visiovox.utils.face_alignment import warp_face_by_landmarks_5_points # Exemplo de import

class FaceSwapper:
    """
    Swaps a face from a source image onto a target face in another image
    using an ONNX model (e.g., Inswapper).
    """

    def __init__(self, config_manager: ConfigManager, resource_manager: ResourceManager, device: str = 'cpu'):
        """
        Initializes the FaceSwapper.

        Args:
            config_manager (ConfigManager): Instance of the configuration manager.
            resource_manager (ResourceManager): Instance of the resource manager.
            device (str): 'cuda' ou 'cpu'.
        """
        self.config_manager: ConfigManager = config_manager
        self.resource_manager: ResourceManager = resource_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.swapper_model: Optional[onnxruntime.InferenceSession] = None
        self.model_name_loaded: Optional[str] = None
        self._input_names: Optional[List[str]] = None # Pode ter múltiplas entradas (ex: target_face, source_embedding)
        self._output_names: Optional[List[str]] = None
        self._input_shape_hw: Optional[Tuple[int, int]] = None # (height, width) para a face de destino no modelo
        self.device = device
        
        # Referência para os 5 pontos de landmark canônicos para alinhamento (112x112 ou 128x128)
        # Estes são os pontos para os quais alinharemos as faces de origem e destino antes da inferência.
        # As coordenadas exatas dependem do template usado no treinamento do inswapper_128.
        # Exemplo para um template 112x112 (precisa ser verificado/ajustado para inswapper_128):
        self.reference_landmarks_5pt_112x112 = np.array([
            [30.2946 + 8.0, 51.6963],  # Olho esquerdo (canto externo)
            [65.5318 + 8.0, 51.5014],  # Olho direito (canto externo)
            [48.0252 + 8.0, 71.7366],  # Ponta do nariz
            [33.5493 + 8.0, 92.3655],  # Canto esquerdo da boca
            [62.7299 + 8.0, 92.2041]   # Canto direito da boca
        ], dtype=np.float32)
        # Se o inswapper_128 usar um template 128x128, esses pontos precisam ser escalados ou um novo conjunto usado.

        self.inswapper_emap = None
        emap_path = os.path.join(
            self.config_manager.get_project_root(),
            'models', 'face_swappers', 'inswapper_128_emap.npy'
        )
        if os.path.exists(emap_path):
            self.inswapper_emap = np.load(emap_path)
            self.logger.info(f"Matriz emap do inswapper_128 carregada de {emap_path}.")
        else:
            self.logger.warning(f"Matriz emap do inswapper_128 não encontrada em {emap_path}. O swap pode ficar borrado.")

        self.logger.info("FaceSwapper initialized.")

    def _get_model_config(self, model_name: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        config_key = f"model_catalog.face_swappers.{model_name}.{key}"
        value = self.config_manager.get(config_key, default)
        if value is None and default is None:
             self.logger.warning(f"Configuration key '{config_key}' not found for swapper model '{model_name}'.")
        return value

    def load_swapping_model(self, model_name: str = "inswapper_128.fp16") -> bool:
        self.logger.info(f"Attempting to load face swapping model: {model_name}")
        if self.swapper_model is not None and self.model_name_loaded == model_name:
            self.logger.info(f"Swapping model '{model_name}' is already loaded.")
            return True

        model_data_from_res_mgr = self.resource_manager.load_model(model_name=model_name, model_type="face_swappers")
        if not model_data_from_res_mgr:
            self.logger.error(f"Failed to get model path/info for swapper '{model_name}' from ResourceManager.")
            return False

        path_relative_to_project_root: Optional[str] = None
        model_details_to_use : Optional[Dict[str, Any]] = None

        if isinstance(model_data_from_res_mgr, dict):
            model_details_to_use = model_data_from_res_mgr
            path_relative_to_project_root = model_details_to_use.get('path_local_cache')
        elif isinstance(model_data_from_res_mgr, str):
            path_relative_to_project_root = model_data_from_res_mgr
            manifest_details = self.resource_manager.get_model_details_from_manifest(model_name, "face_swappers")
            if manifest_details: model_details_to_use = manifest_details
        
        if not path_relative_to_project_root:
            self.logger.error(f"Path_local_cache not found for swapper model '{model_name}'.")
            return False

        try:
            project_root = self.config_manager.get_project_root()
            absolute_model_path = os.path.join(project_root, path_relative_to_project_root)
            
            if not os.path.exists(absolute_model_path):
                self.logger.error(f"ONNX swapper model file not found at: {absolute_model_path}")
                return False

            self.logger.debug(f"Loading ONNX swapper session from: {absolute_model_path}")
            providers = self.config_manager.get("onnx_providers", ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.swapper_model = onnxruntime.InferenceSession(absolute_model_path, providers=providers)
            self.logger.info(f"ONNX Runtime providers ativos: {self.swapper_model.get_providers()}")

            # --- NOVO: extrair emap diretamente do modelo ONNX ---
            try:
                import onnx
                model_onnx = onnx.load(absolute_model_path)
                found_emap = False
                for initializer in model_onnx.graph.initializer:
                    if 'emap' in initializer.name.lower():
                        self.inswapper_emap = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(512, 512)
                        self.logger.info(f"Matriz emap extraída do modelo ONNX: {initializer.name}")
                        found_emap = True
                        break
                if not found_emap:
                    self.logger.warning("Nenhuma matriz emap encontrada no modelo ONNX. O swap pode ficar borrado.")
            except Exception as e:
                self.logger.warning(f"Falha ao tentar extrair emap do modelo ONNX: {e}")

            # --- Fim do bloco de extração emap ---

            # Configurar nomes de entrada/saída e shape
            default_input_names = [inp.name for inp in self.swapper_model.get_inputs()]
            default_output_names = [out.name for out in self.swapper_model.get_outputs()]
            if model_details_to_use:
                self._input_names = model_details_to_use.get("input_names", default_input_names)
                self._output_names = model_details_to_use.get("output_names", default_output_names)
                input_shape_cfg = model_details_to_use.get("input_size_hw")
                if isinstance(input_shape_cfg, list) and len(input_shape_cfg) == 2:
                    self._input_shape_hw = tuple(input_shape_cfg)
            if not self._input_names: self._input_names = default_input_names
            if not self._output_names: self._output_names = default_output_names
            if not self._input_shape_hw:
                for inp in self.swapper_model.get_inputs():
                    if inp.name.lower() == 'target' or (isinstance(inp.shape, list) and len(inp.shape) == 4 and inp.shape[1] == 3):
                        if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                             self._input_shape_hw = (inp.shape[2], inp.shape[3])
                             break
                if not self._input_shape_hw:
                    self._input_shape_hw = (128, 128)
                    self.logger.warning(f"Using default input_shape {self._input_shape_hw} for swapper '{model_name}'.")
            if not (isinstance(self._input_names, list) and len(self._input_names) >= 2):
                 self.logger.error(f"Swapper model '{model_name}' does not have the expected number of inputs. Found: {self._input_names}")
                 self.swapper_model = None
                 return False
            self.model_name_loaded = model_name
            self.logger.info(f"Swapping model '{model_name}' loaded. Inputs: {self._input_names}, Outputs: {self._output_names}, Target Face Input Shape (H,W): {self._input_shape_hw}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load ONNX swapper model '{model_name}': {e}", exc_info=True)
            self.swapper_model = None
            return False

    def load_model(self, model_name: str) -> bool:
        """
        Alias para load_swapping_model, para compatibilidade com o restante do pipeline.
        """
        return self.load_swapping_model(model_name)

    def _select_5_landmarks(self, landmarks_any: np.ndarray) -> Optional[np.ndarray]:
        num_input_landmarks = landmarks_any.shape[0]
        if not isinstance(landmarks_any, np.ndarray) or landmarks_any.ndim != 2 or landmarks_any.shape[1] != 2:
            if hasattr(self, 'logger'):
                self.logger.error(f"Invalid landmarks_any. Expected (N,2) array, got {landmarks_any.shape if isinstance(landmarks_any, np.ndarray) else type(landmarks_any)}")
            return None

        # Índices para os 5 pontos canônicos do MediaPipe FaceMesh (468/478 landmarks)
        mediapipe_5_point_indices = [33, 263, 1, 61, 291]

        if num_input_landmarks >= max(mediapipe_5_point_indices) + 1:
            try:
                selected_landmarks = landmarks_any[mediapipe_5_point_indices, :]
                if hasattr(self, 'logger'):
                    self.logger.info(f"Selected 5 key landmarks using MediaPipe indices: {mediapipe_5_point_indices}")
                return selected_landmarks.astype(np.float32)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error selecting 5 landmarks using MediaPipe indices: {e}", exc_info=True)
                return None
        elif num_input_landmarks == 68:
            # Fallback para 68 pontos (dlib)
            dlib_5_point_indices = [36, 45, 30, 48, 54]  # L-eye, R-eye, Nose, L-mouth, R-mouth
            try:
                selected_landmarks = landmarks_any[dlib_5_point_indices, :]
                if hasattr(self, 'logger'):
                    self.logger.warning("Attempting 5-point selection using dlib 68-point indices as a fallback.")
                return selected_landmarks.astype(np.float32)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Fallback to dlib 68-point indices also failed: {e}", exc_info=True)
                return None
        else:
            if hasattr(self, 'logger'):
                self.logger.error(f"Not enough landmarks ({num_input_landmarks}) provided to select all 5 key points using MediaPipe indices (max index needed: {max(mediapipe_5_point_indices)}).")
            return None

    def _align_face_to_template(self, image_bgr_hwc: np.ndarray, 
                                source_landmarks_5pt: np.ndarray, 
                                target_template_size_hw: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Aligns a face (given by its 5 landmarks) to a canonical template size.
        Returns the warped face and the inverse transformation matrix.

        Args:
            image_bgr_hwc: The image containing the face to be aligned.
            source_landmarks_5pt: The 5 source landmarks (x,y) on the face in image_bgr_hwc.
            target_template_size_hw: The (height, width) of the output aligned face template.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
                - Warped face image (BGR, HWC, uint8) of size target_template_size_hw.
                - Inverse affine transformation matrix (2x3) to warp back.
                Returns (None, None) on failure.
        """
        target_h, target_w = target_template_size_hw
        
        # Usar os pontos de referência canônicos escalados para o target_template_size_hw
        # Se o target_template_size_hw for [128, 128] (comum para inswapper_128)
        # precisamos escalar self.reference_landmarks_5pt_112x112 ou definir um novo template.
        # Vamos definir um template para 128x128 (aproximadamente escalado do de 112x112)
        if target_w == 128 and target_h == 128: # Específico para Inswapper128
             # Scaled reference from typical 112x112 ArcFace points.
             # These are approximate and should match training of inswapper_128
            ref_112 = self.reference_landmarks_5pt_112x112
            scale_factor = 128.0 / 112.0
            dst_pts_template = ref_112 * scale_factor
        elif target_w == 112 and target_h == 112: # Para ArcFace (se o usarmos aqui)
            dst_pts_template = self.reference_landmarks_5pt_112x112
        else:
            self.logger.warning(f"No pre-defined 5-point template for target size {target_template_size_hw}. Using scaled 112x112 ref. Alignment might be suboptimal.")
            # Escalar o template de 112x112
            ref_112 = self.reference_landmarks_5pt_112x112
            scale_ref_w = target_w / 112.0
            scale_ref_h = target_h / 112.0
            dst_pts_template = ref_112 * np.array([scale_ref_w, scale_ref_h], dtype=np.float32)

        try:
            # Usar scikit-image para transformação de similaridade, que é robusta
            from skimage import transform as tf
            tform_sim = tf.SimilarityTransform()
            if not tform_sim.estimate(source_landmarks_5pt, dst_pts_template):
                self.logger.error("Failed to estimate similarity transform for face alignment (swapper).")
                return None, None
            
            # A matriz de transformação direta (para aplicar o warp)
            # A matriz da skimage é 3x3. Precisamos da parte 2x3 para cv2.warpAffine
            forward_matrix = tform_sim.params[:2]

            # A matriz de transformação inversa (para colar de volta)
            # Invertendo a transformação de similaridade completa e pegando a parte afim
            tform_sim_inv = tf.SimilarityTransform(matrix=np.linalg.inv(tform_sim.params))
            inverse_matrix = tform_sim_inv.params[:2]

            warped_face = cv2.warpAffine(image_bgr_hwc, forward_matrix, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)
            self.logger.debug(f"Face aligned for swapper to shape: {warped_face.shape}")
            return warped_face, inverse_matrix

        except ImportError:
            self.logger.error("scikit-image is required for robust face alignment. Please install it: pip install scikit-image")
            return None, None
        except Exception as e:
            self.logger.error(f"Error during _align_face_to_template: {e}", exc_info=True)
            return None, None

    def _preprocess_target_face_for_swapper(self, aligned_target_face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocessa a face de destino alinhada para o modelo de troca de rosto.
        Args:
            aligned_target_face_bgr: Imagem da face de destino alinhada em formato BGR (H, W, 3).
        Returns:
            batch_input: Imagem preprocessada no formato (1, C, H, W) pronta para o modelo.
        """
        img_rgb = aligned_target_face_bgr[..., ::-1]  # BGR para RGB
        img_float = img_rgb.astype(np.float32)
        img_normalized = (img_float - 127.5) / 127.5  # Normaliza para [-1, 1]
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) para (C, H, W)
        batch_input = np.expand_dims(img_chw, axis=0)  # Adiciona dimensão de batch (1, C, H, W)
        
        logger.debug(f"Imagem preprocessada - min: {batch_input.min()}, max: {batch_input.max()}")
        return batch_input

    def _postprocess_swapped_face(self, model_output_nchw: np.ndarray) -> Optional[np.ndarray]:
        """
        Pós-processa a saída do Inswapper128: NCHW->HWC, RGB->BGR, [0,1] ou [-1,1] -> [0,255] uint8
        """
        try:
            if model_output_nchw.ndim == 4 and model_output_nchw.shape[0] == 1:
                output_chw = model_output_nchw[0]
            elif model_output_nchw.ndim == 3:
                output_chw = model_output_nchw
            else:
                self.logger.error(f"Shape inesperado na saída do modelo: {model_output_nchw.shape}")
                return None
            output_hwc_rgb = np.transpose(output_chw, (1, 2, 0))
            # Corrigir faixa de valores igual ao VisoMaster
            if output_hwc_rgb.max() <= 1.0 and output_hwc_rgb.min() >= 0.0:
                output_hwc_rgb_uint8 = np.clip(output_hwc_rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                # fallback: [-1,1] para [0,255]
                output_hwc_rgb_uint8 = np.clip((output_hwc_rgb + 1) * 127.5, 0, 255).astype(np.uint8)
            output_hwc_bgr = cv2.cvtColor(output_hwc_rgb_uint8, cv2.COLOR_RGB2BGR)
            return output_hwc_bgr
        except Exception as e:
            self.logger.error(f"Erro no pós-processamento da face trocada: {e}", exc_info=True)
            return None

    def _convert_embedding_for_inswapper(self, embedding: np.ndarray) -> np.ndarray:
        # Para Inswapper128: apenas normalizar L2 e reshape, igual ao VisoMaster
        n_e = embedding / np.linalg.norm(embedding)
        return n_e.reshape(1, -1).astype(np.float32)

    def _create_soft_oval_mask_numpy(self, h, w, center=None, radius_x=None, radius_y=None, feather=16):
        """
        Cria uma máscara oval suave (float32, [0,1]) com feathering, igual ao VisoMaster.
        """
        if center is None:
            center = (w // 2, h // 2)
        if radius_x is None:
            radius_x = int(w * 0.375)  # 48 para 128x128
        if radius_y is None:
            radius_y = int(h * 0.4375) # 56 para 128x128
        Y, X = np.ogrid[:h, :w]
        dist = ((X - center[0]) / radius_x) ** 2 + ((Y - center[1]) / radius_y) ** 2
        mask = 1 - dist
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        mask = mask.astype(np.float32) / 255.0
        return mask

    def _blend_swapped_face(self, swapped_face_aligned_bgr, original_img_bgr, inv_M, output_shape):
        """
        Cola a face trocada na imagem original usando máscara oval/feathered igual ao VisoMaster.
        """
        h, w = swapped_face_aligned_bgr.shape[:2]
        mask = self._create_soft_oval_mask_numpy(h, w)
        # Salvar máscara para debug
        cv2.imwrite('data/output/debug_oval_mask.jpg', (mask * 255).astype(np.uint8))
        # Warpar máscara para o espaço da imagem original
        mask_warped = cv2.warpAffine(mask, inv_M[:2], (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Warpar face trocada
        swapped_face_warped = cv2.warpAffine(swapped_face_aligned_bgr, inv_M[:2], (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Blending
        mask3 = np.stack([mask_warped]*3, axis=-1)
        blended = (mask3 * swapped_face_warped.astype(np.float32) + (1 - mask3) * original_img_bgr.astype(np.float32)).astype(np.uint8)
        return blended

    def swap_face(self,
                  source_face_embedding: np.ndarray,
                  target_image_bgr_hwc: np.ndarray,
                  target_face_landmarks: np.ndarray
                  ) -> Optional[np.ndarray]:
        """
        Realiza o swap facial completo: alinhamento, preprocess, inferência, pós-processamento e colagem.
        """
        if self.swapper_model is None or self._input_names is None or self._output_names is None or self._input_shape_hw is None:
            self.logger.error("FaceSwapper model not loaded or not properly configured.")
            return None
        if source_face_embedding is None:
            self.logger.error("Source face embedding is None. Cannot swap.")
            return None
        if target_face_landmarks is None:
            self.logger.error("Target face landmarks are None. Cannot swap (needed for alignment).")
            return None

        # 1. Selecionar 5 landmarks
        target_landmarks_5pt = self._select_5_landmarks(target_face_landmarks)
        if target_landmarks_5pt is None:
            self.logger.error("Could not select 5 landmarks from target face.")
            return None

        # 2. Alinhar a face de destino
        aligned_target_face_bgr, inverse_matrix = self._align_face_to_template(
            target_image_bgr_hwc,
            target_landmarks_5pt,
            self._input_shape_hw
        )
        if aligned_target_face_bgr is None or inverse_matrix is None:
            self.logger.error("Target face alignment failed.")
            return None
        cv2.imwrite('data/output/debug_aligned_target_face.jpg', aligned_target_face_bgr)

        # 3. Pré-processar para o modelo
        preprocessed_target_face_batch = self._preprocess_target_face_for_swapper(aligned_target_face_bgr)
        if preprocessed_target_face_batch is None:
            self.logger.error("Preprocessing of aligned target face failed.")
            return None
        # Salvar input do modelo como imagem RGB para debug
        try:
            input_img_rgb = (np.transpose(preprocessed_target_face_batch[0], (1,2,0)) * 255).clip(0,255).astype(np.uint8)
            cv2.imwrite('data/output/debug_input_model_rgb.jpg', cv2.cvtColor(input_img_rgb, cv2.COLOR_RGB2BGR))
            self.logger.info("Input do modelo salvo em data/output/debug_input_model_rgb.jpg")
        except Exception as e:
            self.logger.warning(f"Falha ao salvar input do modelo para debug: {e}")

        # 4. Preparar embedding
        if source_face_embedding.ndim == 1:
            source_face_embedding_batch = self._convert_embedding_for_inswapper(source_face_embedding)
        else:
            source_face_embedding_batch = self._convert_embedding_for_inswapper(source_face_embedding[0])
        # Salvar embedding para debug
        np.save('data/output/debug_embedding_input.npy', source_face_embedding_batch)
        self.logger.info(f"Embedding salvo em data/output/debug_embedding_input.npy. Norma: {np.linalg.norm(source_face_embedding_batch):.4f}")
        # 5. Converter para PyTorch float32 (igual ao VisoMaster)
        torch_device = torch.device(self.device)
        img_tensor = torch.from_numpy(preprocessed_target_face_batch).float()
        emb_tensor = torch.from_numpy(source_face_embedding_batch).float()
        dtype_np = np.float32
        img_tensor = img_tensor.to(torch_device)
        emb_tensor = emb_tensor.to(torch_device)
        output_tensor = torch.empty((1,3,128,128), dtype=img_tensor.dtype, device=torch_device)
        # 6. Inferência via io_binding (igual ao VisoMaster)
        io_binding = self.swapper_model.io_binding()
        io_binding.bind_input(name='target', device_type=self.device, device_id=0, element_type=dtype_np, shape=(1,3,128,128), buffer_ptr=img_tensor.data_ptr())
        io_binding.bind_input(name='source', device_type=self.device, device_id=0, element_type=dtype_np, shape=(1,512), buffer_ptr=emb_tensor.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=dtype_np, shape=(1,3,128,128), buffer_ptr=output_tensor.data_ptr())
        self.swapper_model.run_with_iobinding(io_binding)
        swapped_face_model_size_nchw = output_tensor.cpu().numpy()
        # 7. Pós-processar
        swapped_face_aligned_bgr = self._postprocess_swapped_face(swapped_face_model_size_nchw)
        # Salvar output do modelo antes do blending como imagem RGB para debug
        try:
            output_model_rgb = np.transpose(swapped_face_model_size_nchw[0], (1,2,0))
            output_model_rgb_img = (output_model_rgb * 255).clip(0,255).astype(np.uint8)
            cv2.imwrite('data/output/debug_output_model_rgb.jpg', cv2.cvtColor(output_model_rgb_img, cv2.COLOR_RGB2BGR))
            self.logger.info("Output do modelo salvo em data/output/debug_output_model_rgb.jpg")
        except Exception as e:
            self.logger.warning(f"Falha ao salvar output do modelo para debug: {e}")
        if swapped_face_aligned_bgr is None:
            self.logger.error("Postprocessing of swapped face failed.")
            return None
        cv2.imwrite('data/output/debug_swapped_face_model_output.jpg', swapped_face_aligned_bgr)
        # 8. Cola a face trocada na imagem original usando blending fiel ao VisoMaster
        result_img = self._blend_swapped_face(swapped_face_aligned_bgr, target_image_bgr_hwc, inverse_matrix, target_image_bgr_hwc.shape)
        self.logger.info(f"Face swap concluído. Shape final: {result_img.shape}")
        cv2.imwrite('data/output/debug_final_blended_result.jpg', result_img)
        return result_img

    def preprocess_for_inswapper128_visomaster(self, crop_bgr: np.ndarray, debug_save_path: str = None) -> np.ndarray:
        """
        Pré-processamento idêntico ao VisoMaster para Inswapper128.
        - crop_bgr: imagem alinhada (128x128, BGR, uint8)
        - debug_save_path: se fornecido, salva o tensor de entrada para comparação
        Retorna: tensor (1, 3, 128, 128), float32, normalizado, ordem RGB, [-1, 1]
        """
        # 1. Converter BGR para RGB
        crop_rgb = crop_bgr[:, :, ::-1] # Simples inversão de canais
        # 2. float32
        img = crop_rgb.astype(np.float32)
        # 3. Normalizar para [-1, 1]
        img = (img - 127.5) / 127.5
        # 4. HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # 5. Adicionar batch
        img = np.expand_dims(img, axis=0)
        
        if debug_save_path: # Este debug_save_path não será mais usado diretamente por swap_face
            np.save(debug_save_path, img)
            self.logger.info(f"VisoMaster Preprocess (original [-1,1] path): Imagem de debug salva em {debug_save_path}")

        return img.astype(np.float32)

    def convert_embedding_for_inswapper_visomaster(self, embedding: np.ndarray) -> np.ndarray:
        n_e = embedding / np.linalg.norm(embedding)
        if self.inswapper_emap is not None:
            latent = np.dot(n_e.reshape(1, -1), self.inswapper_emap)
            latent = latent / np.linalg.norm(latent)
            return latent.astype(np.float32)
        else:
            return n_e.reshape(1, -1).astype(np.float32)

    def postprocess_swapped_face_visomaster(self, model_output_nchw: np.ndarray) -> Optional[np.ndarray]:
        try:
            if model_output_nchw.ndim == 4 and model_output_nchw.shape[0] == 1:
                output_chw = model_output_nchw[0]
            elif model_output_nchw.ndim == 3:
                output_chw = model_output_nchw
            else:
                self.logger.error(f"Shape inesperado na saída do modelo: {model_output_nchw.shape}")
                return None
            # VisoMaster: saída geralmente em [0,1] (float32, RGB)
            output_hwc_rgb = np.transpose(output_chw, (1, 2, 0))
            if output_hwc_rgb.max() <= 1.0:
                output_hwc_rgb_uint8 = np.clip(output_hwc_rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                # fallback: [-1,1] para [0,255]
                output_hwc_rgb_uint8 = np.clip((output_hwc_rgb + 1) * 127.5, 0, 255).astype(np.uint8)
            output_hwc_bgr = cv2.cvtColor(output_hwc_rgb_uint8, cv2.COLOR_RGB2BGR)
            return output_hwc_bgr
        except Exception as e:
            self.logger.error(f"Erro no pós-processamento da face trocada: {e}", exc_info=True)
            return None

# --- Placeholder for utility functions that might be moved to visiovox.utils.face_alignment ---
# (Inspired by FaceFusion's face_helper.py and typical alignment procedures)

# def get_reference_landmarks_5pt(template_name: str = 'ffhq_128', image_size_hw: Tuple[int, int] = (128, 128)) -> np.ndarray:
#     """
#     Returns a set of 5 canonical landmark points for a given template and image size.
#     These are the Dst points for warping.
#     """
#     # These are just examples and need to be accurate for the chosen template/model
#     if template_name == 'ffhq_112' and image_size_hw == (112,112):
#         return np.array([ # Standard ArcFace 112x112 reference
#             [30.2946 + 8.0, 51.6963], [65.5318 + 8.0, 51.5014],
#             [48.0252 + 8.0, 71.7366], [33.5493 + 8.0, 92.3655],
#             [62.7299 + 8.0, 92.2041]
#         ], dtype=np.float32)
#     elif template_name == 'ffhq_128' and image_size_hw == (128,128): # Scale of 128/112 = 1.1428
#         ref_112 = get_reference_landmarks_5pt('ffhq_112', (112,112))
#         return ref_112 * (128.0 / 112.0)
#     # Add other templates (ffhq_256, ffhq_512 etc.) or load from a config
#     raise ValueError(f"Unknown landmark template or size: {template_name}, {image_size_hw}")

# Exemplo de uso (para teste isolado, seria chamado pelo Orchestrator)
# if __name__ == '__main__':
#     # Setup
#     from visiovox.core.logger_setup import setup_logging
#     setup_logging()
#     test_logger_swapper = logging.getLogger("face_swapper_main_test")

#     # Mock ConfigManager and ResourceManager
#     # ... (similar to other test blocks, ensure paths and configs for inswapper_128 are mocked) ...

#     try:
#         test_logger_swapper.info("--- FaceSwapper Isolated Test ---")
#         # cfg = MockConfigManagerSwapper()
#         # res = MockResourceManagerSwapper(cfg)
#         # swapper = FaceSwapper(config_manager=cfg, resource_manager=res)
#         # model_loaded = swapper.load_swapping_model(model_name="inswapper_128")
#         # assert model_loaded, "Swapper model failed to load."

#         # TODO: Create dummy source_embedding (e.g. np.random.rand(512).astype(np.float32))
#         # TODO: Create dummy target_image_bgr_hwc (e.g. from file or random)
#         # TODO: Create dummy target_face_landmarks_68pt (plausible coordinates on the target_image)
#         # test_logger_swapper.info("Preparing dummy data for swap_face...")
#         # dummy_embedding = np.random.rand(512).astype(np.float32)
#         # dummy_target_img = np.random.randint(0,255, (480,640,3), dtype=np.uint8)
#         # dummy_target_lms = np.array([[100,100],[150,100],[125,130],[110,160],[140,160]] + [[0,0]]*63, dtype=np.float32) # Mock 68 pts with first 5 being key

#         # result_image = swapper.swap_face(dummy_embedding, dummy_target_img, dummy_target_lms)
#         # if result_image is not None:
#         #     test_logger_swapper.info(f"Face swap successful. Output image shape: {result_image.shape}")
#         #     cv2.imwrite("data/output/swapped_face_test_output.jpg", result_image)
#         #     test_logger_swapper.info("Swapped image saved to data/output/swapped_face_test_output.jpg")
#         # else:
#         #     test_logger_swapper.error("Face swap failed.")
#         test_logger_swapper.warning("FaceSwapper test block needs to be fully implemented with valid dummy data and model.")

#     except Exception as e:
#         test_logger_swapper.critical(f"Error in FaceSwapper test: {e}", exc_info=True)