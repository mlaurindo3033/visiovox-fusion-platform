import yaml
import os
import logging
from typing import Any, Optional, Dict

# Define uma exceção customizada para erros relacionados à configuração
class ConfigError(Exception):
    """Custom exception for configuration related errors."""
    pass

class ConfigManager:
    """
    Manages loading and accessing application configurations from YAML files.
    """
    _instance = None # Singleton instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "configs/default_config.yaml"):
        """
        Initializes the ConfigManager and loads the configuration.
        It's designed as a singleton, so subsequent instantiations
        with different paths will not reload unless explicitly forced or managed.
        For simplicity in this first pass, we assume one main config_path at init.

        Args:
            config_path (str): Path to the main YAML configuration file.
                               Defaults to "configs/default_config.yaml" relative
                               to the project root.
        """
        # To allow re-initialization for testing or specific scenarios if needed,
        # check if already initialized with data.
        if hasattr(self, '_config_data') and self._config_data is not None:
            # Potentially log if trying to re-initialize with a different path,
            # or just return, as per singleton behavior.
            # For now, let it re-evaluate if called directly again.
            pass

        self.logger = logging.getLogger(__name__)
        self._config_data: Optional[Dict] = None
        self.config_file_path: str = config_path

        # Determine the project root directory assuming this file is at src/visiovox/core/config_manager.py
        # This makes the default config_path relative to the project root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        absolute_config_path = os.path.join(project_root, config_path)

        self._load_config(absolute_config_path)

        # Carregar manifesto de modelos se existir
        manifest_path = os.path.join(project_root, "configs", "model_manifest.yaml")
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest_data = yaml.safe_load(mf)
                if manifest_data:
                    self._config_data["model_catalog"] = manifest_data
                    self.logger.info(f"Model manifest loaded from: {manifest_path}")
            else:
                self.logger.info(f"No model manifest found at: {manifest_path}, skipping.")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing model manifest YAML file {manifest_path}: {e}")
        except IOError as e:
            self.logger.error(f"IOError reading model manifest file {manifest_path}: {e}")

        # Adicionar o atributo project_root na inicialização
        self._project_root = self._determine_project_root()
        self.logger.info(f"Project root determined: {self._project_root}")

    def _determine_project_root(self) -> str:
        """Determines the project root directory."""
        # Assume que config_manager.py está em src/visiovox/core/
        # Então, subimos três níveis para chegar à raiz do projeto.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        return project_root

    def get_project_root(self) -> str:
        """Returns the determined project root directory."""
        return self._project_root

    def _load_config(self, absolute_config_path: str) -> None:
        """
        Loads the configuration from the specified YAML file.

        Args:
            absolute_config_path (str): The absolute path to the YAML configuration file.
        
        Raises:
            ConfigError: If the configuration file is not found or is malformed.
        """
        try:
            if not os.path.exists(absolute_config_path):
                self.logger.error(f"Configuration file not found: {absolute_config_path}")
                # RF1.5: Consider raising ConfigError for critical failure
                raise ConfigError(f"Configuration file not found: {absolute_config_path}")
            
            with open(absolute_config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            if self._config_data is None: # Handles empty YAML file case
                self.logger.warning(f"Configuration file is empty: {absolute_config_path}")
                self._config_data = {} # Treat as empty config rather than error
            self.logger.info(f"Configuration loaded successfully from: {absolute_config_path}")

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration file {absolute_config_path}: {e}")
            # RF1.5: Consider raising ConfigError for critical failure
            raise ConfigError(f"Malformed YAML in {absolute_config_path}: {e}") from e
        except IOError as e: # Handles other potential I/O errors
            self.logger.error(f"IOError when trying to read configuration file {absolute_config_path}: {e}")
            raise ConfigError(f"IOError for {absolute_config_path}: {e}") from e

    def get(self, config_key: str, default_value: Any = None) -> Any:
        """
        Retrieves a configuration value for a given key.
        Supports nested keys using dot notation (e.g., "module.submodule.param").

        Args:
            config_key (str): The configuration key to retrieve.
                              Nested keys are separated by dots.
            default_value (Any, optional): The value to return if the key is not found.
                                           Defaults to None.

        Returns:
            Any: The configuration value, or the default_value if not found or if config is not loaded.
        """
        if self._config_data is None:
            self.logger.error("Configuration data is not loaded. Cannot get key.")
            return default_value

        keys = config_key.split('.')
        current_level_data = self._config_data
        
        for key_part in keys:
            if isinstance(current_level_data, dict) and key_part in current_level_data:
                current_level_data = current_level_data[key_part]
            else:
                self.logger.debug(f"Configuration key '{config_key}' (part '{key_part}') not found. Returning default value: {default_value}")
                return default_value
        
        return current_level_data

    # RF1.6 (Opcional): Permitir que valores de configuração YAML sejam sobrescritos por variáveis de ambiente.
    # Esta funcionalidade pode ser adicionada aqui.
    # Exemplo de como poderia ser integrado no método _load_config ou no get:
    # def _apply_env_overrides(self, prefix="VISIOVOX_"):
    #     if self._config_data is None:
    #         return
    #     for env_var, value in os.environ.items():
    #         if env_var.startswith(prefix):
    #             # Convert VISIOVOX_MODULE_SUBMODULE_PARAM to module.submodule.param
    #             config_key_parts = env_var[len(prefix):].lower().split('_')
    #             self.logger.info(f"Configuration key '{'.'.join(config_key_parts)}' overridden by environment variable '{env_var}'. 