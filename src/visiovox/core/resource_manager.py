import logging
from typing import Any, Dict, Optional, Union
import os

# Supondo que ConfigManager está no mesmo pacote 'core'
from .config_manager import ConfigManager 
from visiovox.utils.model_downloader import ModelDownloadManager

class ResourceManager:
    """
    Manages the loading and unloading of resources, particularmente modelos de IA.
    Esta versão inicial simula o carregamento de modelos recuperando caminhos
    da configuração e rastreando-os.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_manager: ConfigManager):
        """
        Inicializa o ResourceManager.

        Args:
            config_manager (ConfigManager): Instância do ConfigManager
                                            para acessar configurações de modelos.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return # Already initialized

        self.config_manager: ConfigManager = config_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        self._loaded_models: Dict[str, Any] = {}  # Armazena modelos "carregados" (caminhos ou placeholders)
        self.model_download_manager = ModelDownloadManager(config_manager=config_manager) # Instanciar aqui
        
        # Determine project root assuming this file is in src/visiovox/core/
        self.project_root: str = config_manager.get_project_root() # Obter do ConfigManager
        
        self._initialized = True
        self.logger.info("ResourceManager initialized.")

    def _get_config(self, key: str, default: Optional[Any] = None) -> Any:
        """Convenience method to get configuration from ConfigManager."""
        return self.config_manager.get(key, default)

    def _get_model_path_from_manifest(self, model_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Tries to get model details (including path) from the model_manifest.yaml via ConfigManager.
        If a download_url is present, it triggers a download via ModelDownloadManager.
        Returns the model details dictionary from the manifest if found and processed, else None.
        """
        self.logger.debug(f"_get_model_path_from_manifest: Searching for model '{model_name}' of type '{model_type}'")
        model_type_key = model_type.replace("_", "") # e.g. face_detector -> facedetector
        
        # Tentativa de usar chaves singulares e plurais para model_type ao buscar no catálogo
        # Ex: 'face_enhancer' e 'face_enhancers'
        # O manifesto está usando singular (ex: face_enhancer), mas o código pode chamar com plural.
        # A padronização para singular no manifesto e no código que chama é a melhor solução a longo prazo.
        # Por enquanto, tentaremos ambos.
        
        # Padronizado para singular no manifesto (e.g., face_enhancer, face_landmarker)
        # Os tipos de modelo nos módulos também foram padronizados para singular.
        model_catalog_key = f"model_catalog.{model_type}" 
        model_type_section = self.config_manager.get(model_catalog_key)

        if not isinstance(model_type_section, dict):
            self.logger.debug(f"_get_model_path_from_manifest: No section for model type '{model_type}' (key: {model_catalog_key}) found in catalog, or it's not a dictionary.")
            return None

        model_info = model_type_section.get(model_name)

        if not isinstance(model_info, dict):
            self.logger.debug(f"_get_model_path_from_manifest: Model '{model_name}' not found under type '{model_type}' in the catalog.")
            return None
        
        # self.logger.debug(f"_get_model_path_from_manifest: Found details for '{model_name}' under type '{model_type}': {model_info}") # Verbose

        # Se o modelo tiver uma URL de download, garantir que ele seja baixado.
        if "download_url" in model_info and "path_local_cache" in model_info and "sha256_hash_expected" in model_info:
            download_successful = self._download_model_from_manifest_if_needed(model_name, model_info)
            if not download_successful:
                self.logger.error(f"Download failed for model '{model_name}' from manifest. Cannot use this model.")
                return None # Indica que o modelo não pôde ser disponibilizado
            self.logger.info(f"_get_model_path_from_manifest: Model '{model_name}' (Type: '{model_type}') downloaded/verified. Path: {os.path.join(self.project_root, model_info['path_local_cache'])}")
            # Retorna o dicionário model_info completo, pois contém o caminho e outros metadados úteis.
            return model_info 
        elif "path_local_cache" in model_info:
            # Se não há URL de download, mas há um caminho local, supõe-se que está pré-disponibilizado.
            self.logger.info(f"_get_model_path_from_manifest: Model '{model_name}' (Type: '{model_type}') found with local path. No download URL provided.")
            # Verificar se o arquivo existe, mesmo sem download_url
            full_local_path = os.path.join(self.project_root, model_info['path_local_cache'])
            if not os.path.exists(full_local_path):
                self.logger.error(f"Model '{model_name}' configured with path_local_cache '{full_local_path}' but file not found and no download_url provided.")
                return None
            return model_info
        else:
            self.logger.warning(f"Model '{model_name}' in manifest is missing 'download_url' or 'path_local_cache' or 'sha256_hash_expected'. Cannot process.")
            return None

    def _download_model_from_manifest_if_needed(self, model_name: str, model_info: Dict[str, Any]) -> bool:
        """
        Helper to download a model defined in the manifest if it's not already present or hash mismatches.
        Returns True if the model is available and correct, False otherwise.
        """
        full_download_path = os.path.join(self.project_root, model_info['path_local_cache'])
        
        # Garante que o diretório de cache local exista
        os.makedirs(os.path.dirname(full_download_path), exist_ok=True)

        return ModelDownloadManager.download_model(
            url=model_info['download_url'],
            dest_path=full_download_path,
            expected_sha256=model_info['sha256_hash_expected']
        )

    def get_model_details_from_manifest(self, model_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full details dictionary for a given model from the manifest.
        Does not trigger download, just fetches metadata.
        """
        self.logger.debug(f"get_model_details_from_manifest: Searching for model '{model_name}' of type '{model_type}'")
        model_catalog_key = f"model_catalog.{model_type}" 
        model_type_section = self.config_manager.get(model_catalog_key)
        
        if isinstance(model_type_section, dict) and model_name in model_type_section:
            model_details = model_type_section.get(model_name)
            if isinstance(model_details, dict):
                # self.logger.debug(f"get_model_details_from_manifest: Found details for '{model_name}' under type '{model_type}': {model_details}") # Verbose
                return model_details
            else:
                self.logger.debug(f"get_model_details_from_manifest: Entry for '{model_name}' under '{model_type}' is not a dictionary.")
        else:
            self.logger.debug(f"get_model_details_from_manifest: No section for model type '{model_type}' or model '{model_name}' not found in catalog.")
            
        return None

    def load_model(self, model_name: str, model_type: str, force_download_check: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Loads a model path, trying the manifest first, then config file, then default path.
        Manages download and verification if specified in the manifest.
        """
        self.logger.info(f"Attempting to load model: Name='{model_name}', Type='{model_type}', ForceCheck='{force_download_check}'")

        if not force_download_check and model_name in self._loaded_models:
             self.logger.debug(f"Model '{model_name}' already in _loaded_models. Returning cached info.")
             # Se já estiver carregado, e não for forçada a verificação, retorna o que foi armazenado.
             # O que é armazenado pode ser o dict de detalhes do manifesto ou um caminho de string.
             return self._loaded_models[model_name]

        # 1. Try to load from model_manifest.yaml
        # _get_model_path_from_manifest retorna o dict model_info se bem sucedido
        model_manifest_details = self._get_model_path_from_manifest(model_name, model_type)
        if model_manifest_details:
            self.logger.info(f"Model '{model_name}' (Type: '{model_type}') sourced via manifest.")
            self._loaded_models[model_name] = model_manifest_details # Cache the full details
            return model_manifest_details # Retorna o dicionário de detalhes

        # 2. Fallback: Try to get path from default_config.yaml (legacy or non-manifest models)
        self.logger.debug(f"Model '{model_name}' (Type: '{model_type}') not found or failed to process via manifest. Trying config fallback...")
        config_model_path = self._get_model_path_from_config(model_name, model_type)
        if config_model_path:
            if os.path.exists(config_model_path):
                self.logger.info(f"Model '{model_name}' (Type: '{model_type}') found via config at: {config_model_path}")
                self._loaded_models[model_name] = config_model_path # Cache o caminho
                return config_model_path # Retorna o caminho como string
            else:
                self.logger.warning(f"Model '{model_name}' (Type: '{model_type}') configured in config but not found at: {config_model_path}")
        
        # 3. Fallback: Special handling for yolo_default_onnx if not in manifest/config
        # Este é um fallback específico e pode ser removido se o yolo for sempre via manifesto/config.
        if model_name == "yolo_default_onnx" and model_type == "face_detection":
            self.logger.debug(f"Trying default path for '{model_name}' as a last resort...")
            default_yolo_path = os.path.join(self.project_root, "models", "face_detection", "yolov8n-face.onnx")
            if os.path.exists(default_yolo_path):
                self.logger.info(f"Model '{model_name}' (Type: '{model_type}') found at default hardcoded path: {default_yolo_path}")
                self._loaded_models[model_name] = default_yolo_path # Cache o caminho
                return default_yolo_path # Retorna o caminho como string
            else:
                self.logger.warning(f"Default YOLO model '{model_name}' not found at {default_yolo_path}")

        self.logger.error(f"Failed to load model '{model_name}' (Type: '{model_type}'). Not found in manifest, config, or default paths.")
        return None

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Directly access a configuration value from the ConfigManager.
        """
        return self.config_manager.get(key, default)

    def _get_model_path_from_config(self, model_name: str, model_type: str) -> Optional[str]:
        """
        Attempts to get the model path directly from default_config.yaml (legacy).
        Constructs a key like "models.face_detection.yolo_default_onnx.path".
        """
        # Construct the configuration key based on model type and name
        # Example: "models.face_detection.yolo_default_onnx.path"
        # Ensure model_type is a simple string, e.g., "face_detection" (pluralization might be needed if types are singular)
        config_key = f"models.{model_type}.{model_name}.path"
        
        model_path_rel = self.config_manager.get(config_key)
        
        if model_path_rel:
            absolute_model_path = os.path.join(self.project_root, model_path_rel)
            # self.logger.debug(f"Path for '{model_name}' from config: '{absolute_model_path}'")
            return absolute_model_path
        else:
            # self.logger.debug(f"Path for '{model_name}' (key: '{config_key}') not found in config.")
            return None

    def unload_model(self, model_name: str) -> bool:
        """
        Descarrega um modelo.
        Nesta versão placeholder, remove o modelo do dicionário rastreado.

        Args:
            model_name (str): Nome do modelo a ser descarregado.

        Returns:
            bool: True se o modelo foi encontrado e "descarregado", False caso contrário.
        """
        self.logger.info(f"Attempting to unload model: '{model_name}'")
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            self.logger.debug(f"Model '{model_name}' successfully unloaded from cache.")
            return True
        else:
            self.logger.warning(f"Model '{model_name}' not found in loaded models cache. Cannot unload.")
            return False

    def get_loaded_model_info(self, model_name: str) -> Optional[Any]:
        """
        Retrieves the representation of an already loaded model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[Any]: The model representation if loaded, else None.
        """
        # self.logger.debug(f"Attempting to get loaded model info for: '{model_name}'") # Can be verbose
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        self.logger.debug(f"Model '{model_name}' not currently in ResourceManager's loaded cache.")
        return None

    def _construct_default_path(self, model_name: str, model_type: str, extension: str = ".onnx") -> str:
        # ... existing code ...
        return self.model_download_manager.download_model_if_needed(
            url=model_info['download_url'],
            download_path=full_download_path,
            expected_hash=model_info['sha256_hash_expected'],
            model_name=model_name
        )

# Exemplo de uso (seria em outro lugar da aplicação):
# if __name__ == '__main__':
#     from .config_manager import ConfigManager  # Assume está no mesmo diretório para teste
#     import logging
#     logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#
#     # Certifique-se que 'configs/default_config.yaml' existe e tem a estrutura:
#     # models:
#     #   face_detection:
#     #     yolo_model_1:
#     #       path: "path/to/yolo_model_1.onnx"
#     #   text_generation:
#     #     gpt_model_small:
#     #       path: "path/to/gpt_model_small.bin"
#
#     try:
#         cfg = ConfigManager(config_path="configs/default_config.yaml")  # Ajuste o caminho se necessário
#         res_manager = ResourceManager(config_manager=cfg)
#
#         # Test load_model
#         model_path1 = res_manager.load_model(model_name="yolo_model_1", model_type="face_detection")
#         print(f"Loaded yolo_model_1 path: {model_path1}")
#         model_path_cached = res_manager.load_model(model_name="yolo_model_1", model_type="face_detection")
#         print(f"Cached yolo_model_1 path: {model_path_cached}")
#
#         model_path2 = res_manager.load_model(model_name="gpt_model_small", model_type="text_generation")
#         print(f"Loaded gpt_model_small path: {model_path2}")
#         
#         non_existent_model = res_manager.load_model("non_existent", "some_type")
#         print(f"Non_existent_model: {non_existent_model}")
#
#         # Test unload_model
#         print(f"Unloading yolo_model_1: {res_manager.unload_model('yolo_model_1')}")
#         print(f"Unloading yolo_model_1 again: {res_manager.unload_model('yolo_model_1')}")
#         print(f"Unloading gpt_model_small: {res_manager.unload_model('gpt_model_small')}")
#
#     except Exception as e:
#         print(f"An error occurred during ResourceManager test: {e}") 