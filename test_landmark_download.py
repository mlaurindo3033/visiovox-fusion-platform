import sys, os

# Adiciona src/ ao path para importar o pacote visiovox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager

# Modelos de face_landmarker para testar
models = [
    ('2dfan4', 'face_landmarkers'),
    ('fan_68_5', 'face_landmarkers'),
]

if __name__ == '__main__':
    cfg = ConfigManager()
    rm = ResourceManager(cfg)

    for name, mtype in models:
        path = rm.load_model(name, mtype)
        print(f"{name} ({mtype}): {path}")
        if path and os.path.exists(path):
            print(f"SUCCESS: '{name}' baixado com sucesso em '{path}'")
        else:
            print(f"ERROR: falha ao baixar ou arquivo não existe: '{path}'") 