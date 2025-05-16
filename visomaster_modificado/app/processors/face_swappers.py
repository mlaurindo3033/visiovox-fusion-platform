import torch
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
import onnx
from typing import TYPE_CHECKING, List, Tuple, Dict, Optional, Union
import os
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.helpers.downloader import download_file
from app.helpers.miscellaneous import is_file_exists
import traceback
class FaceSwappers:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        if not self.models_processor.models[arcface_model]:
            self.models_processor.models[arcface_model] = self.models_processor.load_model(arcface_model)

        if arcface_model == 'CSCSArcFace':
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(arcface_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image
        
    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    # Versão em lote para reconhecimento de múltiplos rostos
    def run_recognize_batch(self, 
                          img: torch.Tensor, 
                          kps_list: List[np.ndarray], 
                          similarity_type: str = 'Opal', 
                          arcface_model: str = 'Inswapper128ArcFace') -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Processa múltiplos rostos em um único lote para otimizar o uso da GPU.
        
        Args:
            img: Imagem de entrada
            kps_list: Lista de keypoints para cada rosto
            similarity_type: Tipo de similaridade
            arcface_model: Modelo ArcFace a ser usado
            
        Returns:
            Tuple contendo listas de embeddings e imagens recortadas
        """
        if not kps_list:
            return [], []
            
        if not self.models_processor.models[arcface_model]:
            self.models_processor.models[arcface_model] = self.models_processor.load_model(arcface_model)
            
        if arcface_model == 'CSCSArcFace':
            # Para CSCS, chamamos a função individual para cada rosto pois o processamento é específico
            embeddings = []
            cropped_images = []
            for kps in kps_list:
                embedding, cropped_image = self.recognize_cscs(img, kps)
                embeddings.append(embedding)
                cropped_images.append(cropped_image)
            return embeddings, cropped_images
        else:
            # Para outros modelos, processamos em lote
            return self.recognize_batch(arcface_model, img, kps_list, similarity_type)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        """
        Método revisado de reconhecimento usando template fixo para maior robustez.
        """
        # Simplificar o tratamento de keypoints
        face_kps_array = np.array(face_kps, dtype=np.float32)
        
        # Tratamento robusto para garantir formato (5,2)
        if face_kps_array.ndim == 1 and len(face_kps_array) >= 10:
            # Array plano [x1,y1,x2,y2,...] -> reshape para [[x1,y1],[x2,y2],...]
            face_kps_array = face_kps_array[:10].reshape(5, 2)
        elif face_kps_array.ndim == 2:
            # Se já for 2D mas com mais de 5 pontos, usar apenas os primeiros 5
            if face_kps_array.shape[0] > 5:
                face_kps_array = face_kps_array[:5]
            # Se tiver 5 pontos mas mais de 2 coordenadas por ponto, usar apenas x,y
            if face_kps_array.shape[1] > 2:
                face_kps_array = face_kps_array[:, :2]
                
        # Verificação final do formato
        if face_kps_array.shape != (5, 2):
            raise ValueError(f"Falha ao converter keypoints para formato (5,2). Forma atual: {face_kps_array.shape}")
        
        # Template fixo (pontos faciais padronizados para arcface)
        template = np.array([
            [38.2946, 51.6963],  # olho esquerdo
            [73.5318, 51.5014],  # olho direito
            [56.0252, 71.7366],  # nariz
            [41.5493, 92.3655],  # canto esquerdo da boca
            [70.7299, 92.2041]   # canto direito da boca
        ], dtype=np.float32)
        
        # Inicializar transformação
        tform = trans.SimilarityTransform()
        
        try:
            # Estimar transformação com template fixo
            tform.estimate(face_kps_array, template)
            
            # Aplicar transformação
            aligned_face = v2.functional.affine(img, 
                                             tform.rotation*57.2958, 
                                             (tform.translation[0], tform.translation[1]), 
                                             tform.scale, 
                                             0, 
                                             center=(0,0))
            aligned_face = v2.functional.crop(aligned_face, 0, 0, 112, 112)
            cropped_image = aligned_face.permute(1, 2, 0).clone()  # 112, 112, 3
            
            # Normalização específica para cada modelo
            if arcface_model == 'Inswapper128ArcFace':
                if aligned_face.dtype == torch.uint8:
                    aligned_face = aligned_face.to(torch.float32)
                aligned_face = torch.sub(aligned_face, 127.5)
                aligned_face = torch.div(aligned_face, 127.5)
            elif arcface_model == 'SimSwapArcFace':
                if aligned_face.dtype == torch.uint8:
                    aligned_face = torch.div(aligned_face.to(torch.float32), 255.0)
                aligned_face = v2.functional.normalize(aligned_face, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
            else:
                if aligned_face.dtype == torch.uint8:
                    aligned_face = aligned_face.to(torch.float32)
                aligned_face = torch.div(aligned_face, 127.5)
                aligned_face = torch.sub(aligned_face, 1)
                
            # Debug
            print(f"Reconhecimento bem-sucedido com template fixo - modelo: {arcface_model}")
                
            # Preparar para inferência
            aligned_face = torch.unsqueeze(aligned_face, 0).contiguous()
            input_name = self.models_processor.models[arcface_model].get_inputs()[0].name
            
            outputs = self.models_processor.models[arcface_model].get_outputs()
            output_names = []
            for o in outputs:
                output_names.append(o.name)
            
            io_binding = self.models_processor.models[arcface_model].io_binding()
            io_binding.bind_input(
                name=input_name, 
                device_type=self.models_processor.device, 
                device_id=0, 
                element_type=np.float32,  
                shape=aligned_face.size(), 
                buffer_ptr=aligned_face.data_ptr()
            )
            
            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.models_processor.device)
            
            # Sincronizar e executar modelo
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
                
            self.models_processor.models[arcface_model].run_with_iobinding(io_binding)
            
            # Verificar saída
            outputs = io_binding.copy_outputs_to_cpu()
            if not outputs or len(outputs) == 0:
                raise ValueError(f"Nenhuma saída gerada pelo modelo {arcface_model}")
            
            # Retornar embedding e imagem recortada
            embedding = np.array(outputs[0]).flatten()
            # Dump automático do embedding para debug/comparação
            try:
                debug_dir = r"D:\visiovox-fusion-platform\data\visomaster_npy_intermediarios"
                os.makedirs(debug_dir, exist_ok=True)
                np.save(os.path.join(debug_dir, "arcface_embedding_dump.npy"), embedding)
                print(f"[DEBUG] Embedding ArcFace salvo em {os.path.join(debug_dir, 'arcface_embedding_dump.npy')}. Norma: {np.linalg.norm(embedding):.4f}")
            except Exception as e:
                print(f"[DEBUG] Falha ao salvar embedding ArcFace: {e}")
            # Salvar input alinhado para debug/comparação
            try:
                debug_dir = r"D:\visiovox-fusion-platform\data\visomaster_npy_intermediarios"
                os.makedirs(debug_dir, exist_ok=True)
                np.save(os.path.join(debug_dir, "arcface_input_aligned.npy"), aligned_face.cpu().numpy())
                print(f"[DEBUG] Input alinhado ArcFace salvo em {os.path.join(debug_dir, 'arcface_input_aligned.npy')}. Shape: {aligned_face.shape}, Dtype: {aligned_face.dtype}")
            except Exception as e:
                print(f"[DEBUG] Falha ao salvar input alinhado ArcFace: {e}")
            return embedding, cropped_image
            
        except Exception as e:
            print(f"ERRO CRÍTICO no reconhecimento {arcface_model}: {str(e)}")
            print(f"Keypoints: {face_kps_array}")
            raise

    def recognize_batch(self, arcface_model, img, face_kps_list, similarity_type):
        """
        Processa múltiplos rostos em lote para reconhecimento facial.
        """
        batch_size = len(face_kps_list)
        if batch_size == 0:
            return [], []

        # Preparar tensores para o lote
        batch_imgs = []
        cropped_images = []

        # Obter o template de destino
        if similarity_type == 'Optimal':
            dst = self.models_processor.arcface_dst
        else:
            dst = faceutil.get_arcface_template(image_size=112, mode='arcface')
            dst = np.squeeze(dst)

        # Processa cada rosto para preparar o tensor normalizado
        for face_kps in face_kps_list:
            # Verificação básica dos keypoints
            if face_kps is None or len(face_kps) == 0:
                print("Aviso: Keypoints inválidos neste rosto, pulando")
                continue
                
            # Garantir formato correto
            face_kps_array = np.array(face_kps)
            
            # Verificar e corrigir formato se necessário
            if face_kps_array.shape != (5, 2):
                # Se for array 1D, tentar remodelar
                if len(face_kps_array.shape) == 1 and face_kps_array.size >= 10:
                    face_kps_array = face_kps_array.reshape(-1, 2)
                
                # Garantir que temos 5 pontos
                if len(face_kps_array) != 5:
                    if len(face_kps_array) < 5:
                        print(f"Aviso: Número insuficiente de pontos faciais: {len(face_kps_array)}")
                        continue
                    face_kps_array = face_kps_array[:5]
            
            # Tentar estimar a transformação mantendo o formato original
            try:
                if similarity_type == 'Optimal':
                    tform = trans.SimilarityTransform()
                    try:
                        # Primeiro tentar com o formato original completo
                        tform.estimate(face_kps_array, dst)
                    except Exception as e:
                        print(f"Aviso: Erro ao usar formato original do template em lote: {str(e)}")
                        # APENAS como fallback, tentar formato simplificado
                        if len(dst.shape) == 3 and dst.shape[0] == 5 and dst.shape[1] == 5:
                            dst_5_by_2 = np.array([dst[i][0] for i in range(5)])
                            print("ATENÇÃO: Usando formato simplificado (5,2) como FALLBACK em lote - pode afetar qualidade")
                            tform.estimate(face_kps_array, dst_5_by_2)
                        else:
                            print(f"Falha completa em lote, pulando rosto")
                            continue

                    temp = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]), tform.scale, 0, center=(0,0))
                    temp = v2.functional.crop(temp, 0, 0, 112, 112)
                    cropped_image = temp.permute(1, 2, 0).clone()  # 112,112,3
                    if temp.dtype == torch.uint8:
                        temp = torch.div(temp.to(torch.float32), 255.0)
                    temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                    processed_img = temp
                else:
                    tform = trans.SimilarityTransform()
                    try:
                        # Primeiro tentar com o formato original completo
                        tform.estimate(face_kps_array, dst)
                    except Exception as e:
                        print(f"Aviso: Erro ao usar formato original do template em lote: {str(e)}")
                        # APENAS como fallback, tentar formato simplificado
                        if len(dst.shape) == 3 and dst.shape[0] == 5 and dst.shape[1] == 5:
                            dst_5_by_2 = np.array([dst[i][0] for i in range(5)])
                            print("ATENÇÃO: Usando formato simplificado (5,2) como FALLBACK em lote - pode afetar qualidade")
                            tform.estimate(face_kps_array, dst_5_by_2)
                        else:
                            print(f"Falha completa em lote, pulando rosto")
                            continue

                    processed_img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]), tform.scale, 0, center=(0,0))
                    processed_img = v2.functional.crop(processed_img, 0, 0, 112, 112)
                    
                # Normalização específica do modelo
                if arcface_model == 'Inswapper128ArcFace':
                    cropped_image = processed_img.permute(1, 2, 0).clone()
                    if processed_img.dtype == torch.uint8:
                        processed_img = processed_img.to(torch.float32)
                    processed_img = torch.sub(processed_img, 127.5)
                    processed_img = torch.div(processed_img, 127.5)
                elif arcface_model == 'SimSwapArcFace':
                    cropped_image = processed_img.permute(1, 2, 0).clone()
                    if processed_img.dtype == torch.uint8:
                        processed_img = torch.div(processed_img.to(torch.float32), 255.0)
                    processed_img = v2.functional.normalize(processed_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
                else:
                    cropped_image = processed_img.permute(1, 2, 0).clone()
                    if processed_img.dtype == torch.uint8:
                        processed_img = processed_img.to(torch.float32)
                    processed_img = torch.div(processed_img, 127.5)
                    processed_img = torch.sub(processed_img, 1)

                batch_imgs.append(processed_img)
                cropped_images.append(cropped_image)
                
            except Exception as e:
                print(f"Erro ao processar rosto em lote: {str(e)}")
                # Pular este rosto em caso de erro
                continue

        # Monta o tensor do lote [batch_size, channels, height, width]
        if not batch_imgs:
            return [], []  # Retornar listas vazias se não houver nenhum rosto válido
            
        batch_tensor = torch.stack(batch_imgs)

        # Prepara para inferência em lote
        input_name = self.models_processor.models[arcface_model].get_inputs()[0].name
        outputs = self.models_processor.models[arcface_model].get_outputs()
        output_names = [o.name for o in outputs]

        io_binding = self.models_processor.models[arcface_model].io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=batch_tensor.size(),
            buffer_ptr=batch_tensor.data_ptr()
        )

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], self.models_processor.device)

        # Sincroniza e executa o modelo
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

        self.models_processor.models[arcface_model].run_with_iobinding(io_binding)

        # Obtém os embeddings (saídas do modelo)
        outputs_cpu = io_binding.copy_outputs_to_cpu()
        
        # Processa os embeddings para o formato esperado
        embeddings = []
        for i in range(len(batch_imgs)):
            # Extrair embedding individual do lote
            if len(outputs_cpu[0].shape) == 2:  # Formato típico para embeddings em lote [batch_size, embedding_dim]
                embedding = outputs_cpu[0][i]
            else:
                # Se o formato for diferente, adapte conforme necessário
                embedding = outputs_cpu[0].flatten()
                
            embeddings.append(embedding)

        return embeddings, cropped_images

    def preprocess_image_cscs(self, img, face_kps):
        """
        Pré-processa a imagem para o modelo CSCS, seguindo a lógica do código original.
        
        Args:
            img: Imagem de entrada (tensor)
            face_kps: Keypoints faciais
            
        Returns:
            Tuple contendo a imagem processada e a imagem recortada
        """
        try:
            # Converter face_kps para array numpy se necessário
            face_kps_array = np.array(face_kps, dtype=np.float32)
            
            # Verificar formato dos keypoints
            if face_kps_array.shape != (5, 2):
                if len(face_kps_array.shape) == 1 and len(face_kps_array) >= 10:
                    face_kps_array = face_kps_array[:10].reshape(5, 2)
                elif face_kps_array.shape[0] > 5:
                    face_kps_array = face_kps_array[:5]
                if face_kps_array.shape != (5, 2):
                    raise ValueError(f"Formato de keypoints inválido: {face_kps_array.shape}")
            
            # Usar transformação de similaridade com template FFHQ_kps
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps_array, self.models_processor.FFHQ_kps)
            
            # Aplicar transformação para obter face 512x512
            temp = v2.functional.affine(
                img, 
                tform.rotation*57.2958, 
                (tform.translation[0], tform.translation[1]), 
                tform.scale, 
                0, 
                center=(0,0),
                interpolation=v2.InterpolationMode.BILINEAR
            )
            temp = v2.functional.crop(temp, 0, 0, 512, 512)
            
            # Redimensionar para 112x112 como no código original
            image = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)(temp)
            
            # Preservar imagem recortada para visualização
            cropped_image = image.permute(1, 2, 0).clone()
            
            # Normalizar para [-1, 1]
            if image.dtype == torch.uint8:
                image = torch.div(image.to(torch.float32), 255.0)
            
            image = v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
            
            # Retornar imagem processada e imagem recortada
            return torch.unsqueeze(image, 0).contiguous(), cropped_image
            
        except Exception as e:
            print(f"ERRO CRÍTICO no pré-processamento CSCS: {str(e)}")
            print(f"Keypoints: {face_kps_array}")
            raise
    
    def recognize_cscs(self, img, face_kps):
        """
        Versão revisada do reconhecimento CSCS seguindo a lógica do código original.
        """
        try:
            # Pré-processar a imagem
            img, cropped_image = self.preprocess_image_cscs(img, face_kps)
            
            # Garantir que o modelo está carregado
            if not self.models_processor.models['CSCSArcFace']:
                self.models_processor.models['CSCSArcFace'] = self.models_processor.load_model('CSCSArcFace')
                
            # Criar binding para o modelo
            io_binding = self.models_processor.models['CSCSArcFace'].io_binding()
            io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
            io_binding.bind_output(name='output', device_type=self.models_processor.device)
            
            # Sincronizar e executar
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
                
            self.models_processor.models['CSCSArcFace'].run_with_iobinding(io_binding)
            
            # Processar a saída
            output = io_binding.copy_outputs_to_cpu()[0]
            embedding = torch.from_numpy(output).to('cpu')
            embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
            embedding = embedding.numpy().flatten()
            
            # Adicionar embedding do adaptador de ID, como no código original
            try:
                embedding_id = self.recognize_cscs_id_adapter(img, None)
                embedding = embedding + embedding_id
            except Exception as e:
                print(f"Aviso: Não foi possível adicionar embedding_id: {str(e)}")
            
            return embedding, cropped_image
            
        except Exception as e:
            print(f"ERRO CRÍTICO no reconhecimento CSCS: {str(e)}")
            print(traceback.format_exc())
            raise

    def recognize_cscs_id_adapter(self, img, _):
        """
        Método para obter o embedding do adaptador de ID para o modelo CSCS.
        
        Args:
            img: Imagem de entrada (tensor)
            _: Parâmetro ignorado para manter compatibilidade com a interface
            
        Returns:
            Embedding do adaptador de ID
        """
        try:
            # Garantir que o modelo está carregado
            if not self.models_processor.models['CSCSIDArcFace']:
                self.models_processor.models['CSCSIDArcFace'] = self.models_processor.load_model('CSCSIDArcFace')
                
            # Criar binding para o modelo
            io_binding = self.models_processor.models['CSCSIDArcFace'].io_binding()
            io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
            io_binding.bind_output(name='output', device_type=self.models_processor.device)
            
            # Sincronizar e executar
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
                
            self.models_processor.models['CSCSIDArcFace'].run_with_iobinding(io_binding)
            
            # Processar a saída
            output = io_binding.copy_outputs_to_cpu()[0]
            embedding = torch.from_numpy(output).to('cpu')
            embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
            embedding = embedding.numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Aviso: Erro no adaptador de ID CSCS: {str(e)}")
            # Retornar array de zeros como fallback
            return np.zeros(512, dtype=np.float32)

    def calc_inswapper_latent(self, source_embedding):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def calc_inswapper_latent_batch(self, source_embeddings):
        """
        Processa múltiplos embeddings em lote para o Inswapper.
        
        Args:
            source_embeddings: Lista de embeddings de origem
            
        Returns:
            Lista de embeddings processados
        """
        latents = []
        for embedding in source_embeddings:
            n_e = embedding / l2norm(embedding)
            latent = n_e.reshape((1, -1))
            latent = np.dot(latent, self.models_processor.emap)
            latent /= np.linalg.norm(latent)
            latents.append(latent)
        return latents

    def run_inswapper(self, image, embedding, output):
        if not self.models_processor.models['Inswapper128']:
            self.models_processor.models['Inswapper128'] = self.models_processor.load_model('Inswapper128')

        # Garantir que o embedding seja um tensor PyTorch
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).to(self.models_processor.device)
        elif isinstance(embedding, torch.Tensor):
            embedding_tensor = embedding.to(self.models_processor.device)
        else:
            raise TypeError(f"Tipo de embedding não suportado: {type(embedding)}")

        # Salvar intermediários para debug (caminho absoluto no projeto principal)
        try:
            debug_output_dir = r"D:\visiovox-fusion-platform\data\visomaster_npy_intermediarios"
            
            print(f"VisoMaster DEBUG (singular): Tentando criar/usar pasta de debug: {debug_output_dir}")
            
            if not os.path.exists(debug_output_dir):
                os.makedirs(debug_output_dir, exist_ok=True)
                print(f"VisoMaster DEBUG (singular): Pasta criada: {debug_output_dir}")
            else:
                print(f"VisoMaster DEBUG (singular): Pasta já existe: {debug_output_dir}")

            target_tensor_path = os.path.join(debug_output_dir, "visomaster_target_input_singular.npy")
            source_embedding_path = os.path.join(debug_output_dir, "visomaster_source_embedding_singular.npy")

            np.save(target_tensor_path, image.cpu().numpy()) # image já é um tensor
            np.save(source_embedding_path, embedding_tensor.cpu().numpy())
            print(f"VisoMaster DEBUG (singular): Salvo target input em {target_tensor_path}")
            print(f"VisoMaster DEBUG (singular): Salvo source embedding em {source_embedding_path}")
        except Exception as e:
            print(f"VisoMaster DEBUG (singular): ERRO ao salvar intermediários: {str(e)}")


        io_binding = self.models_processor.models['Inswapper128'].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding_tensor.data_ptr()) # Usar embedding_tensor
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Inswapper128'].run_with_iobinding(io_binding)

    # Nova função para processar lotes de rostos com Inswapper
    def run_inswapper_batch(self, 
                          images: List[torch.Tensor], 
                          embeddings: List[np.ndarray], 
                          outputs: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        Processa múltiplos rostos em lote com o modelo Inswapper.
        
        Args:
            images: Lista de imagens de destino
            embeddings: Lista de embeddings de origem
            outputs: Lista opcional de tensores de saída pré-alocados
            
        Returns:
            Lista de imagens processadas
        """
        batch_size = len(images)
        if batch_size == 0:
            return []
            
        if not self.models_processor.models['Inswapper128']:
            self.models_processor.models['Inswapper128'] = self.models_processor.load_model('Inswapper128')
            
        # Se outputs não foi fornecido, criar novos tensores
        if outputs is None:
            outputs = []
            for _ in range(batch_size):
                out = torch.empty((1, 3, 128, 128), dtype=torch.float32, device=torch.device(self.models_processor.device)).contiguous()
                outputs.append(out)
                
        # Processar cada imagem em sequência (usando o mesmo modelo carregado)
        # Idealmente, processaríamos tudo em um único lote, mas alguns modelos ONNX
        # podem não suportar operações em lote eficientemente
        for i in range(batch_size):
            io_binding = self.models_processor.models['Inswapper128'].io_binding()
            
            # Garantir que embeddings[i] seja um tensor PyTorch
            current_embedding = embeddings[i]
            if isinstance(current_embedding, np.ndarray):
                embedding_tensor_batch_item = torch.from_numpy(current_embedding).to(self.models_processor.device)
            elif isinstance(current_embedding, torch.Tensor):
                embedding_tensor_batch_item = current_embedding.to(self.models_processor.device)
            else:
                raise TypeError(f"Tipo de embedding no lote não suportado: {type(current_embedding)}")

            # Salvar intermediários para debug (caminho absoluto no projeto principal)
            try:
                debug_output_dir = r"D:\visiovox-fusion-platform\data\visomaster_npy_intermediarios"

                print(f"VisoMaster DEBUG (batch): Tentando criar/usar pasta de debug: {debug_output_dir}")
                
                if not os.path.exists(debug_output_dir):
                    os.makedirs(debug_output_dir, exist_ok=True)
                    print(f"VisoMaster DEBUG (batch): Pasta criada: {debug_output_dir}")
                else:
                    print(f"VisoMaster DEBUG (batch): Pasta já existe: {debug_output_dir}")
                
                target_tensor_path = os.path.join(debug_output_dir, f"visomaster_target_input_batch_{i}.npy")
                source_embedding_path = os.path.join(debug_output_dir, f"visomaster_source_embedding_batch_{i}.npy")
                
                np.save(target_tensor_path, images[i].cpu().numpy())
                np.save(source_embedding_path, embedding_tensor_batch_item.cpu().numpy())
                
                print(f"VisoMaster DEBUG (batch): Salvo target input em {target_tensor_path}")
                print(f"VisoMaster DEBUG (batch): Salvo source embedding em {source_embedding_path}")
            except Exception as e:
                print(f"VisoMaster DEBUG (batch): ERRO ao salvar intermediários para o item {i}: {str(e)}")

            io_binding.bind_input(
                name='target', 
                device_type=self.models_processor.device, 
                device_id=0, 
                element_type=np.float32, 
                shape=(1, 3, 128, 128), 
                buffer_ptr=images[i].data_ptr()
            )
            io_binding.bind_input(
                name='source', 
                device_type=self.models_processor.device, 
                device_id=0, 
                element_type=np.float32, 
                shape=(1, 512), 
                buffer_ptr=embedding_tensor_batch_item.data_ptr() # Usar embedding_tensor_batch_item
            )
            io_binding.bind_output(
                name='output', 
                device_type=self.models_processor.device, 
                device_id=0, 
                element_type=np.float32, 
                shape=(1, 3, 128, 128), 
                buffer_ptr=outputs[i].data_ptr()
            )
            
            # Execute a inferência para este item do lote
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
                
            self.models_processor.models['Inswapper128'].run_with_iobinding(io_binding)
            
        return outputs

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_iss_swapper(self, image, embedding, output, version="A"):
        ISS_MODEL_NAME = f'InStyleSwapper256 Version {version}'
        if not self.models_processor.models[ISS_MODEL_NAME]:
            self.models_processor.models[ISS_MODEL_NAME] = self.models_processor.load_model(ISS_MODEL_NAME)
        
        io_binding = self.models_processor.models[ISS_MODEL_NAME].io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[ISS_MODEL_NAME].run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        #latent /= np.linalg.norm(latent)
        latent = latent/np.linalg.norm(latent,axis=1,keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.models_processor.models['SimSwap512']:
            self.models_processor.models['SimSwap512'] = self.models_processor.load_model('SimSwap512')

        io_binding = self.models_processor.models['SimSwap512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['SimSwap512'].run_with_iobinding(io_binding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        ghostfaceswap_model, output_name = None, None
        if swapper_model == 'GhostFace-v1':
            if not self.models_processor.models['GhostFacev1']:
                self.models_processor.models['GhostFacev1'] = self.models_processor.load_model('GhostFacev1')

            ghostfaceswap_model = self.models_processor.models['GhostFacev1']
            output_name = '781'

        elif swapper_model == 'GhostFace-v2':
            if not self.models_processor.models['GhostFacev2']:
                self.models_processor.models['GhostFacev2'] = self.models_processor.load_model('GhostFacev2')

            ghostfaceswap_model = self.models_processor.models['GhostFacev2']
            output_name = '1165'

        elif swapper_model == 'GhostFace-v3':
            if not self.models_processor.models['GhostFacev3']:
                self.models_processor.models['GhostFacev3'] = self.models_processor.load_model('GhostFacev3')

            ghostfaceswap_model = self.models_processor.models['GhostFacev3']
            output_name = '1549'

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name=output_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_cscs(self, source_embedding):
        """
        Prepara o embedding facial para o modelo CSCS.
        
        Args:
            source_embedding: Embedding facial de origem
            
        Returns:
            Embedding formatado para o modelo CSCS
        """
        # Simplificando a operação de reshape
        latent = source_embedding.reshape(1, -1)
        
        # Garantir que seja um array contíguo na memória para melhor performance
        if not latent.flags['C_CONTIGUOUS']:
            latent = np.ascontiguousarray(latent)
            
        return latent

    def run_swapper_cscs(self, image, embedding, output):
        """
        Executa troca de rosto usando o modelo CSCS.
        
        Args:
            image: Imagem de entrada normalizada (tensor)
            embedding: Embedding facial (numpy array)
            output: Tensor de saída pré-alocado
        """
        # Carregar o modelo se necessário
        if not self.models_processor.models['CSCS']:
            self.models_processor.models['CSCS'] = self.models_processor.load_model('CSCS')

        # Converter embedding para tensor se necessário
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).to(self.models_processor.device)
        else:
            embedding_tensor = embedding

        # IMPORTANTE: O modelo CSCS tem múltiplas saídas, mas precisamos apenas da primeira (output)
        # As outras saídas são tensores intermediários que podem causar artefatos visuais
        
        # Usar run_model em vez de io_binding para ter mais controle sobre as saídas
        try:
            # Preparar as entradas
            ort_inputs = {
                'input_1': image.cpu().numpy(),
                'input_2': embedding_tensor.cpu().numpy() if isinstance(embedding_tensor, torch.Tensor) else embedding_tensor
            }
            
            # Executar o modelo e obter apenas o primeiro output
            ort_outputs = self.models_processor.models['CSCS'].run(['output'], ort_inputs)
            
            # Copiar o resultado para o tensor de saída
            output_np = ort_outputs[0]  # Apenas o primeiro output
            output_tensor = torch.from_numpy(output_np).to(self.models_processor.device)
            output.copy_(output_tensor)
            
            print(f"CSCS - Inferência bem-sucedida. Shape: {output_np.shape}, Min: {output_np.min()}, Max: {output_np.max()}")
            
        except Exception as e:
            print(f"Erro na inferência do CSCS: {str(e)}")
            # Fallback para o método original usando io_binding
            print("Usando método de fallback com io_binding")
            io_binding = self.models_processor.models['CSCS'].io_binding()
            
            # Vincular entradas e saída
            io_binding.bind_input(
                name='input_1',
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=image.shape,
                buffer_ptr=image.data_ptr()
            )
            
            io_binding.bind_input(
                name='input_2',
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=embedding_tensor.shape,
                buffer_ptr=embedding_tensor.data_ptr()
            )
            
            io_binding.bind_output(
                name='output',  # Apenas o primeiro output
                device_type=self.models_processor.device,
                device_id=0,
                element_type=np.float32,
                shape=output.shape,
                buffer_ptr=output.data_ptr()
            )
            
            # Sincronizar e executar
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
                
            self.models_processor.models['CSCS'].run_with_iobinding(io_binding)

    # Método para processar múltiplos rostos em paralelo usando CUDA Streams
    def process_faces_with_streams(self, 
                                 images: List[torch.Tensor], 
                                 embeddings: List[np.ndarray], 
                                 face_mode: str = 'Inswapper',
                                 num_streams: int = 2) -> List[torch.Tensor]:
        """
        Processa múltiplos rostos em paralelo usando streams CUDA para máximo throughput.
        
        Args:
            images: Lista de imagens de destino
            embeddings: Lista de embeddings de origem
            face_mode: Modo de troca de face ('Inswapper', 'SimSwap', 'GhostFace-v2', 'CSCS')
            num_streams: Número de streams CUDA para usar
            
        Returns:
            Lista de imagens processadas
        """
        # Verificar parâmetros de entrada
        if not images or not embeddings:
            return []
            
        if len(images) != len(embeddings):
            raise ValueError(f"O número de imagens ({len(images)}) deve ser igual ao número de embeddings ({len(embeddings)})")
            
        batch_size = len(images)
        
        # Se não estiver usando CUDA, reverte para processamento sequencial
        if self.models_processor.device != "cuda":
            print(f"Dispositivo não é CUDA, usando processamento sequencial para {batch_size} rostos")
            
            if face_mode == 'Inswapper':
                return self.run_inswapper_batch(images, embeddings)
            else:
                # Processa sequencialmente para outros modos
                outputs = []
                for i in range(batch_size):
                    output = torch.empty_like(images[i])
                    try:
                        if face_mode == 'SimSwap':
                            self.run_swapper_simswap512(images[i], embeddings[i], output)
                        elif face_mode.startswith('GhostFace'):
                            self.run_swapper_ghostface(images[i], embeddings[i], output, face_mode)
                        elif face_mode == 'CSCS':
                            self.run_swapper_cscs(images[i], embeddings[i], output)
                        else:
                            raise ValueError(f"Modo de face desconhecido: {face_mode}")
                            
                        outputs.append(output)
                    except Exception as e:
                        print(f"Erro no processamento do rosto {i}: {str(e)}")
                        # Continuar com o próximo, mas adicionar None para manter o índice
                        outputs.append(None)
                        
                # Filtrar resultados None
                return [o for o in outputs if o is not None]
                
        # Para CUDA, usar streams para paralelização
        results = [None] * batch_size
        streams = []
        
        try:
            # Limitar streams ao mínimo entre num_streams e batch_size
            max_streams = min(num_streams, batch_size, torch.cuda.device_count() * 2)
            
            print(f"Processando {batch_size} rostos usando {max_streams} streams CUDA")
            
            # Criar streams CUDA
            for _ in range(max_streams):
                stream = torch.cuda.Stream()
                streams.append(stream)
                
            # Pré-alocar tensores de saída para todos os itens do lote
            outputs = []
            for i in range(batch_size):
                # Criar tensor de saída com o mesmo tamanho da imagem de entrada
                # mas garantir que esteja no mesmo dispositivo
                output = torch.empty_like(images[i])
                outputs.append(output)
            
            # Dividir o lote entre os streams
            tasks = []  # Lista para rastrear todas as tarefas
            for i in range(batch_size):
                stream_idx = i % len(streams)
                stream = streams[stream_idx]
                
                # Executa inferência no stream específico
                with torch.cuda.stream(stream):
                    try:
                        # Seleciona o modelo apropriado baseado no face_mode
                        if face_mode == 'Inswapper':
                            self.run_inswapper(images[i], embeddings[i], outputs[i])
                        elif face_mode == 'SimSwap':
                            self.run_swapper_simswap512(images[i], embeddings[i], outputs[i])
                        elif face_mode.startswith('GhostFace'):
                            self.run_swapper_ghostface(images[i], embeddings[i], outputs[i], face_mode)
                        elif face_mode == 'CSCS':
                            self.run_swapper_cscs(images[i], embeddings[i], outputs[i])
                        else:
                            raise ValueError(f"Modo de face desconhecido: {face_mode}")
                        
                        # Armazenar o resultado
                        results[i] = outputs[i]
                    except Exception as e:
                        print(f"Erro no stream {stream_idx} para rosto {i}: {str(e)}")
                        # Inserir None como marcador de erro
                        results[i] = None
                        
                    tasks.append((i, stream))
            
            # Sincronizar todos os streams
            for i, stream in tasks:
                try:
                    stream.synchronize()
                except Exception as e:
                    print(f"Erro ao sincronizar stream para rosto {i}: {str(e)}")
            
            # Garantir que o dispositivo espere todas as operações antes de continuar
            torch.cuda.synchronize()
                
            # Filtrar resultados None
            valid_results = [r for r in results if r is not None]
            
            print(f"Processamento concluído: {len(valid_results)}/{batch_size} rostos com sucesso")
            return valid_results
            
        except Exception as e:
            print(f"Erro durante processamento em streams CUDA: {str(e)}")
            # Em caso de erro, sincronizar todos os streams antes de sair
            for stream in streams:
                try:
                    stream.synchronize()
                except:
                    pass
                    
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Limpar memória GPU
            
            # Retorna resultados parciais ou processamento sequencial como fallback
            valid_results = [r for r in results if r is not None]
            if valid_results:
                return valid_results
            else:
                print("Fallback para processamento sequencial após erro em streams")
                return self.run_inswapper_batch(images, embeddings)

    def run_swapper_simswap512_batch(self, images: List[torch.Tensor], embeddings: List[np.ndarray], outputs: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        batch_size = len(images)
        if batch_size == 0:
            return []
        if not self.models_processor.models['SimSwap512']:
            self.models_processor.models['SimSwap512'] = self.models_processor.load_model('SimSwap512')
        if outputs is None:
            outputs = [torch.empty((1, 3, 512, 512), dtype=torch.float32, device=torch.device(self.models_processor.device)).contiguous() for _ in range(batch_size)]
        for i in range(batch_size):
            io_binding = self.models_processor.models['SimSwap512'].io_binding()
            io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=images[i].data_ptr())
            io_binding.bind_input(name='onnx::Gemm_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embeddings[i].data_ptr())
            io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=outputs[i].data_ptr())
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['SimSwap512'].run_with_iobinding(io_binding)
        return outputs

    def run_swapper_ghostface_batch(self, images: List[torch.Tensor], embeddings: List[np.ndarray], outputs: Optional[List[torch.Tensor]] = None, swapper_model='GhostFace-v2') -> List[torch.Tensor]:
        batch_size = len(images)
        if batch_size == 0:
            return []
        if swapper_model == 'GhostFace-v1':
            model_name = 'GhostFacev1'
            output_name = '781'
        elif swapper_model == 'GhostFace-v2':
            model_name = 'GhostFacev2'
            output_name = '1165'
        elif swapper_model == 'GhostFace-v3':
            model_name = 'GhostFacev3'
            output_name = '1549'
        else:
            raise ValueError(f"Swapper model desconhecido: {swapper_model}")
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(model_name)
        if outputs is None:
            outputs = [torch.empty((1, 3, 256, 256), dtype=torch.float32, device=torch.device(self.models_processor.device)).contiguous() for _ in range(batch_size)]
        for i in range(batch_size):
            io_binding = self.models_processor.models[model_name].io_binding()
            io_binding.bind_input(name='target', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=images[i].data_ptr())
            io_binding.bind_input(name='source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embeddings[i].data_ptr())
            io_binding.bind_output(name=output_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=outputs[i].data_ptr())
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models[model_name].run_with_iobinding(io_binding)
        return outputs

    def run_swapper_cscs_batch(self, images: List[torch.Tensor], embeddings: List[np.ndarray], outputs: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        batch_size = len(images)
        if batch_size == 0:
            return []
        if not self.models_processor.models['CSCS']:
            self.models_processor.models['CSCS'] = self.models_processor.load_model('CSCS')
        if outputs is None:
            outputs = [torch.empty((1, 3, 256, 256), dtype=torch.float32, device=torch.device(self.models_processor.device)).contiguous() for _ in range(batch_size)]
        for i in range(batch_size):
            io_binding = self.models_processor.models['CSCS'].io_binding()
            io_binding.bind_input(name='input_1', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=images[i].data_ptr())
            io_binding.bind_input(name='input_2', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embeddings[i].data_ptr())
            io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=outputs[i].data_ptr())
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['CSCS'].run_with_iobinding(io_binding)
        return outputs