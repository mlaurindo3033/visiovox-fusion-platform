# VisioVox Fusion Platform

Plataforma de fusão de funcionalidades de visão computacional para detecção, aprimoramento e troca de faces.

## Gerenciamento e Download de Modelos

### Visão Geral
Nosso pipeline de IA carrega modelos sob demanda, verificando seu hash SHA256 para garantir integridade.
- Os metadados de cada modelo ficam em `configs/model_manifest.yaml`.
- O `ConfigManager` injeta esse manifesto em `config_manager.get('model_catalog')`.
- O `ResourceManager.load_model()` delega ao `ModelDownloadManager`, que cria diretórios, baixa o `.onnx` e verifica o hash antes de salvar.

### Estrutura do `configs/model_manifest.yaml`
```yaml
face_swappers:
  inswapper_128:
    description: "InsightFace Swapper (128x128)"
    type: "face_swapper"
    download_url: "<URL .onnx>"
    path_local_cache: "models/face_swappers/inswapper_128.onnx"
    sha256_hash_expected: "<HASH SHA256 completo>"

  simswap_256:
    description: "SimSwap Face Swapper (256x256 input)"
    type: "face_swapper"
    download_url: "<URL .onnx>"
    path_local_cache: "models/face_swappers/simswap_256.onnx"
    sha256_hash_expected: "<HASH SHA256 completo>"

face_enhancers:
  gfpgan_1_4:
    description: "GFPGAN v1.4 - High-fidelity Face Enhancer"
    type: "face_enhancer"
    download_url: "<URL .onnx>"
    path_local_cache: "models/face_enhancers/gfpgan_1_4.onnx"
    sha256_hash_expected: "<HASH SHA256 completo>"

  codeformer:
    description: "CodeFormer - Robust Face Restoration and Enhancement"
    type: "face_enhancer"
    download_url: "<URL .onnx>"
    path_local_cache: "models/face_enhancers/codeformer.onnx"
    sha256_hash_expected: "<HASH SHA256 completo>"
```

### Como Funciona
1. **Cache local**: Se `path_local_cache` existe, retorna o caminho local.
2. **Download sob demanda**: Caso contrário, o `ResourceManager` solicita ao `ModelDownloadManager`, que:
   - Cria o diretório: `os.makedirs(os.path.dirname(path), exist_ok=True)`
   - Baixa o `.onnx` via `requests`.
   - Calcula SHA256 (`hashlib.sha256`) e compara com `sha256_hash_expected`.
   - Salva o arquivo somente se a verificação for bem-sucedida.
3. **Falha segura**: Em caso de erro de rede ou hash mismatch, registra erro e retorna `None`.

### Adicionando Novos Modelos
1. Encontre a URL direta do arquivo `.onnx` no repositório.
2. Calcule o SHA256 completo:
   - Linux/macOS: `sha256sum meu_modelo.onnx`
   - Windows (PowerShell): `Get-FileHash -Algorithm SHA256 .\meu_modelo.onnx | Format-List`
3. No `configs/model_manifest.yaml`, adicione uma entrada sob o `model_type` correto:
   - `download_url`: URL do `.onnx`
   - `path_local_cache`: caminho relativo em `models/`
   - `sha256_hash_expected`: hash obtido
   - `description` e `type`.
4. Teste:
   ```python
   from src.visiovox.core.config_manager import ConfigManager
   from src.visiovox.core.resource_manager import ResourceManager

   cfg = ConfigManager()
   rm = ResourceManager(cfg)
   print(rm.load_model("nome_do_modelo", "model_type"))
   ```

### Próximos Passos
- Documentar contribuições e o processo de adição de novos modelos.
- Integrar módulos de Landmarking, Segmentação ou Enhancers ao pipeline. 
