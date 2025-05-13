import logging
from typing import Any, Dict, Optional

# Supondo que ConfigManager está no mesmo pacote 'core'
from .config_manager import ConfigManager 

class ResourceManager:
    """
    Manages the loading and unloading of resources, particularmente modelos de IA.
    Esta versão inicial simula o carregamento de modelos recuperando caminhos
    da configuração e rastreando-os.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializa o ResourceManager.

        Args:
            config_manager (ConfigManager): Instância do ConfigManager
                                            para acessar configurações de modelos.
        """
        self.config_manager: ConfigManager = config_manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        self._loaded_models: Dict[str, Any] = {}  # Armazena modelos "carregados" (caminhos ou placeholders)
        self.logger.info("ResourceManager initialized.")

    def load_model(self, model_name: str, model_type: str) -> Optional[Any]:
        """
        Carrega um modelo com base em seu nome e tipo.
        Nesta versão placeholder, recupera o caminho do modelo
        na configuração e o retorna como representação do modelo carregado.
        Também faz cache deste modelo "carregado" para simular o processo real.

        Args:
            model_name (str): Nome específico do modelo a ser carregado (ex.: "yolo_default_onnx").
            model_type (str): Tipo do modelo (ex.: "face_detection").
                              Ajuda a estruturar a busca na configuração.

        Returns:
            Optional[Any]: Representação do modelo carregado (ex.: seu caminho ou objeto placeholder),
                           ou None se a configuração do modelo não for encontrada.
        """
        self.logger.info(f"Attempting to load model: Name='{model_name}', Type='{model_type}'")

        if model_name in self._loaded_models:
            self.logger.debug(f"Model '{model_name}' already loaded. Returning cached instance.")
            return self._loaded_models[model_name]

        # Constroi a chave de configuração para encontrar o caminho do modelo
        # ex.: "models.face_detection.yolo_default_onnx.path"
        config_key_path = f"models.{model_type}.{model_name}.path"
        model_path = self.config_manager.get(config_key_path)

        if model_path is None:
            self.logger.error(
                f"Configuration for model path not found for Name='{model_name}', Type='{model_type}' "
                f"(checked key: '{config_key_path}'). Cannot load model."
            )
            return None
        
        # Aqui, o carregamento real do modelo ocorreria em uma implementação real.
        # Por enquanto, a "representação do modelo" é apenas seu caminho.
        # Futuro: carregar sessão ONNX, modelo PyTorch, etc.
        # Exemplo:
        # if model_path.endswith(".onnx"):
        #     import onnxruntime
        #     try:
        #         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        #         absolute_model_path = os.path.join(project_root, model_path)
        #         model_representation = onnxruntime.InferenceSession(absolute_model_path)
        #         self.logger.info(f"ONNX model '{model_name}' loaded from {absolute_model_path}")
        #     except Exception as e:
        #         self.logger.error(f"Failed to load ONNX model '{model_name}' from {model_path}: {e}")
        #         return None
        # else:
        #     self.logger.warning(f"Model type for path '{model_path}' not recognized for direct loading by ResourceManager. Returning path.")
        
        model_representation = model_path  # Placeholder: representação do modelo é seu caminho
        
        self._loaded_models[model_name] = model_representation
        self.logger.info(f"Model '{model_name}' (Type: '{model_type}') 'loaded'. Representation: {model_representation}")
        
        return model_representation

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
            self.logger.debug(f"Model '{model_name}' successfully unloaded.")
            return True
        else:
            self.logger.warning(f"Model '{model_name}' not found in loaded models. Cannot unload.")
            return False

    def get_loaded_model_info(self, model_name: str) -> Optional[Any]:
        """
        Retrieves the representation of an already loaded model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[Any]: The model representation if loaded, else None.
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        self.logger.debug(f"Model '{model_name}' not currently loaded according to ResourceManager.")
        return None

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