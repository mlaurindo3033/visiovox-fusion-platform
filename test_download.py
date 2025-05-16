import sys
import os

# Adiciona src/ ao path para importar o pacote visiovox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager

if __name__ == '__main__':
    cfg = ConfigManager()
    rm = ResourceManager(cfg)
    path = rm.load_model('inswapper_128', 'face_swappers')
    print('Downloaded Path:', path) 