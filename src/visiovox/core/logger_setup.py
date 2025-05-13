import logging
import logging.config
import yaml
import os
from typing import Union

def setup_logging(
    config_path: str = "configs/logging_config.yaml",
    default_level: Union[int, str] = logging.INFO
) -> None:
    """
    Configures the Python logging system using a YAML configuration file.

    If the configuration file is not found or is invalid, it sets up
    a basic console logger with the specified default_level.

    Args:
        config_path (str): Path to the YAML logging configuration file,
                           relative to the project root.
        default_level (Union[int, str]): The default logging level (e.g., logging.INFO, "INFO")
                                         to use if the configuration file cannot be loaded.
    """
    logger = logging.getLogger(__name__)  # Logger for this setup module itself

    # Determine the project root directory assuming this file is at src/visiovox/core/logger_setup.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    absolute_config_path = os.path.join(project_root, config_path)

    if os.path.exists(absolute_config_path):
        try:
            with open(absolute_config_path, 'rt', encoding='utf-8') as f:
                logging_config_dict = yaml.safe_load(f)

            if logging_config_dict:
                # Ensure the 'logs' directory exists if a FileHandler points there.
                for handler_name, handler_config in logging_config_dict.get('handlers', {}).items():
                    if 'filename' in handler_config:
                        log_dir = os.path.dirname(os.path.join(project_root, handler_config['filename']))
                        if not os.path.exists(log_dir):
                            try:
                                os.makedirs(log_dir)
                                logger.info(f"Log directory created: {log_dir}")
                            except OSError as e:
                                # Fallback to basic config if log directory creation fails
                                print(f"Warning: Could not create log directory {log_dir}. Error: {e}. Falling back to basic logging.")
                                logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                                logger.warning(
                                    f"Failed to create log directory for handler '{handler_name}'. Logging configured with basicConfig at level {default_level}."
                                )
                                return  # Exit after basicConfig

                logging.config.dictConfig(logging_config_dict)
                logger.info(f"Logging configured successfully from: {absolute_config_path}")
            else:
                logger.warning(f"Logging configuration file is empty: {absolute_config_path}. Falling back to basicConfig.")
                logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        except yaml.YAMLError as e:
            logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logger.warning(
                f"Error parsing YAML logging configuration file {absolute_config_path}: {e}. Falling back to basicConfig at level {default_level}."
            )
        except Exception as e:
            logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logger.warning(
                f"An unexpected error occurred while configuring logging from {absolute_config_path}: {e}. Falling back to basicConfig at level {default_level}."
            )
    else:
        logging.basicConfig(level=default_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.warning(
            f"Logging configuration file not found: {absolute_config_path}. Falling back to basicConfig at level {default_level}."
        )

# Exemplo de como esta função seria chamada no início da aplicação
# if __name__ == '__main__':
#     setup_logging()  # Usa o caminho e nível padrão
#     main_logger = logging.getLogger("visiovox.main_app")  # Exemplo de logger da aplicação
#     main_logger.debug("Este é um log de debug da aplicação.")
#     main_logger.info("Este é um log de info da aplicação.")
#     main_logger.warning("Este é um log de warning da aplicação.")
#     main_logger.error("Este é um log de error da aplicação.")
#     main_logger.critical("Este é um log critical da aplicação.")
#     another_module_logger = logging.getLogger("visiovox.another_module")
#     another_module_logger.info("Log de outro módulo.")
#     # import requests
#     # try:
#     #     requests.get("http://nonexistentdomain12345.com")
#     # except requests.exceptions.RequestException as e:
#     #     pass  # requests já loga seu erro, que será pego pelo root. 