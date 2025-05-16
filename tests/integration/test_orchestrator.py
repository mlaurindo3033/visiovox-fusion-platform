import sys, os
import cv2
import numpy as np

# Adiciona src/ ao path para importar o pacote visiovox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from visiovox.core.config_manager import ConfigManager
from visiovox.core.resource_manager import ResourceManager
from visiovox.core.orchestrator import Orchestrator

if __name__ == '__main__':
    # Setup
    cfg = ConfigManager()
    rm = ResourceManager(cfg)
    orch = Orchestrator(cfg, rm)

    # For testing, fake face detection to ensure pipeline proceeds
    orch.face_detector.detect_faces = lambda image: [(100, 100, 200, 200)]

    # Create dummy input image
    dummy_path = 'dummy_orchestrator_test.jpg'
    img = np.full((480, 640, 3), 200, dtype=np.uint8)  # Cinza claro para melhor visualização
    cv2.imwrite(dummy_path, img)

    # Remove old output if exists
    out_rel = cfg.get('output_paths.detected_faces_image')
    out_abs = os.path.join(os.getcwd(), out_rel)
    if os.path.exists(out_abs):
        os.remove(out_abs)

    # Run pipeline
    success = orch.process_static_image(dummy_path)
    print(f"Orchestrator pipeline success: {success}")
    if success and os.path.exists(out_abs):
        print(f"Output file created successfully at: {out_abs}")
        print("Please manually inspect the output image to verify drawings.")
    else:
        print(f"Failed to create output file at: {out_abs}") 