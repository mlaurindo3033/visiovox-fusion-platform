import torch
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Union

class CudaStreamPool:
    """
    Gerencia um pool de CUDA streams para processamento paralelo eficiente em GPU.
    
    Esta classe permite a execução de tarefas em streams CUDA separados,
    facilitando o paralelismo na GPU e melhorando o desempenho em operações
    que podem ser executadas concorrentemente.
    """
    
    def __init__(self, num_streams: int = 4, device: str = "cuda"):
        """
        Inicializa o pool de streams CUDA.
        
        Args:
            num_streams (int): Número de streams CUDA a serem criados no pool
            device (str): Dispositivo CUDA a ser usado
        """
        self.device = device
        self.num_streams = num_streams
        self.streams = []
        self.stream_lock = threading.RLock()
        self.stream_in_use = {}
        
        # Verifica se CUDA está disponível
        if not torch.cuda.is_available() and device == "cuda":
            print("AVISO: CUDA não está disponível. Usando CPU para processamento.")
            self.device = "cpu"
            return
            
        # Cria os streams se o dispositivo for CUDA
        if self.device == "cuda":
            for i in range(num_streams):
                stream = torch.cuda.Stream()
                self.streams.append(stream)
                self.stream_in_use[i] = False
    
    def get_stream(self, timeout: float = 1.0) -> Optional[torch.cuda.Stream]:
        """
        Obtém um stream disponível do pool com mecanismo de timeout.
        
        Args:
            timeout: Tempo máximo (em segundos) para esperar por um stream disponível
        
        Returns:
            Um stream CUDA disponível ou None se todos estiverem em uso após timeout
        """
        if self.device != "cuda":
            return None, None
            
        # Primeiro, tenta obter um stream imediatamente
        with self.stream_lock:
            for i, stream in enumerate(self.streams):
                if not self.stream_in_use[i]:
                    self.stream_in_use[i] = True
                    return i, stream
        
        # Se não conseguir imediatamente e timeout > 0, espera e tenta novamente
        if timeout > 0:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Pequena pausa para não sobrecarregar a CPU
                time.sleep(0.01)
                
                # Tenta novamente obter um stream
                with self.stream_lock:
                    for i, stream in enumerate(self.streams):
                        if not self.stream_in_use[i]:
                            self.stream_in_use[i] = True
                            return i, stream
        
        # Se chegou aqui, não conseguiu obter um stream dentro do timeout
        print("Aviso: Todos os streams CUDA estão em uso após timeout. Criando stream temporário.")
        
        # Cria um stream temporário como último recurso
        try:
            temp_stream = torch.cuda.Stream()
            return -1, temp_stream  # Índice -1 indica que é um stream temporário
        except Exception as e:
            print(f"Erro ao criar stream temporário: {str(e)}")
            return None, None
            
    def release_stream(self, stream_idx: int):
        """
        Libera um stream para uso por outras operações.
        
        Args:
            stream_idx (int): Índice do stream a ser liberado
        """
        if self.device != "cuda" or stream_idx < 0:  # Streams temporários têm índice -1
            return
            
        with self.stream_lock:
            if 0 <= stream_idx < len(self.streams):
                # Verificar se o stream ainda está sendo usado antes de liberar
                if self.stream_in_use[stream_idx]:
                    self.stream_in_use[stream_idx] = False
                else:
                    print(f"Aviso: Tentativa de liberar stream {stream_idx} que já está livre")
                    
    def wait_for_any_stream(self, timeout: float = 5.0) -> bool:
        """
        Espera até que qualquer stream esteja disponível ou até que o timeout expire.
        
        Args:
            timeout: Tempo máximo (em segundos) para esperar
            
        Returns:
            True se um stream foi liberado, False se o timeout expirou
        """
        if self.device != "cuda":
            return True
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.stream_lock:
                # Verifica se há algum stream livre
                for i in range(len(self.streams)):
                    if not self.stream_in_use[i]:
                        return True
            
            # Pequena pausa antes de verificar novamente
            time.sleep(0.05)
            
        # Se chegou aqui, o timeout expirou
        return False
    
    def run_on_stream(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executa uma função em um stream CUDA disponível.
        
        Args:
            func (Callable): Função a ser executada no stream
            *args: Argumentos posicionais para a função
            **kwargs: Argumentos nomeados para a função
            
        Returns:
            Resultado da função
        """
        if self.device != "cuda":
            # Se não estiver usando CUDA, apenas executa a função normalmente
            return func(*args, **kwargs)
            
        stream_idx, stream = self.get_stream()
        if stream is None:
            # Se todos os streams estiverem em uso, cria um temporário
            with torch.cuda.Stream() as temp_stream:
                with torch.cuda.stream(temp_stream):
                    result = func(*args, **kwargs)
                temp_stream.synchronize()
            return result
            
        try:
            # Executa a função no stream obtido
            with torch.cuda.stream(stream):
                result = func(*args, **kwargs)
            stream.synchronize()
            return result
        finally:
            # Libera o stream independentemente do resultado
            self.release_stream(stream_idx)
    
    def process_batch(self, 
                    func: Callable, 
                    items: List[Any], 
                    batch_size: Optional[int] = None, 
                    timeout: float = 1.0,
                    **kwargs) -> List[Any]:
        """
        Processa um lote de itens em paralelo usando streams CUDA.
        
        Args:
            func (Callable): Função a ser aplicada a cada item
            items (List[Any]): Lista de itens a serem processados
            batch_size (Optional[int]): Tamanho do lote para processamento
            timeout (float): Tempo máximo para esperar por um stream livre
            **kwargs: Argumentos adicionais para a função
            
        Returns:
            Lista de resultados processados
        """
        if not items:
            return []
            
        results = [None] * len(items)
        
        if self.device != "cuda":
            # Processamento sequencial sem CUDA
            for i, item in enumerate(items):
                results[i] = func(item, **kwargs)
            return results
        
        # Define o tamanho do lote
        if batch_size is None:
            batch_size = min(len(items), self.num_streams * 2)
        
        # Processamento em paralelo com CUDA streams
        temp_streams = []  # Lista para rastrear streams temporários
        submitted_tasks = []
        
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                batch_futures = []
                
                for j, item in enumerate(batch):
                    # Tenta obter um stream com timeout
                    stream_idx, stream = self.get_stream(timeout=timeout)
                    
                    if stream is None:
                        # Se falhar completamente em obter um stream, espere até que algum seja liberado
                        if not self.wait_for_any_stream(timeout=timeout*2):
                            print("Aviso: Não foi possível obter um stream após espera. Usando processamento sequencial para este item.")
                            # Processa este item sem stream
                            results[i + j] = func(item, **kwargs)
                            continue
                        
                        # Tenta novamente após espera
                        stream_idx, stream = self.get_stream(timeout=timeout)
                        if stream is None:
                            # Ainda não conseguiu, processa sequencialmente
                            results[i + j] = func(item, **kwargs)
                            continue
                    
                    # Rastreia se este é um stream temporário
                    if stream_idx == -1:
                        temp_streams.append(stream)
                    
                    # Executa a função no stream
                    with torch.cuda.stream(stream):
                        try:
                            result = func(item, **kwargs)
                            results[i + j] = result
                        except Exception as e:
                            print(f"Erro ao processar item {i+j}: {str(e)}")
                            # Em caso de erro, tenta processar sem stream
                            try:
                                results[i + j] = func(item, **kwargs)
                            except Exception as e2:
                                print(f"Erro ao processar item {i+j} sem stream: {str(e2)}")
                    
                    batch_futures.append((stream_idx, stream))
                
                # Adiciona as tarefas submetidas à lista
                submitted_tasks.extend(batch_futures)
                
                # Se temos muitas tarefas pendentes, espera que algumas terminem
                if len(submitted_tasks) >= self.num_streams * 2:
                    for _ in range(min(self.num_streams, len(submitted_tasks))):
                        if submitted_tasks:
                            idx, stream = submitted_tasks.pop(0)
                            stream.synchronize()
                            self.release_stream(idx)
            
            # Espera todas as tarefas restantes terminarem
            for idx, stream in submitted_tasks:
                stream.synchronize()
                self.release_stream(idx)
            
            # Sincroniza e limpa qualquer stream temporário
            for temp_stream in temp_streams:
                temp_stream.synchronize()
            
            return results
            
        except Exception as e:
            print(f"Erro durante o processamento em lote: {str(e)}")
            
            # Em caso de erro, tenta limpar todos os recursos
            try:
                # Libera todos os streams normais em uso
                for idx, stream in submitted_tasks:
                    if idx >= 0:  # Não é um stream temporário
                        try:
                            stream.synchronize()
                            self.release_stream(idx)
                        except:
                            pass
                
                # Sincroniza streams temporários
                for stream in temp_streams:
                    try:
                        stream.synchronize()
                    except:
                        pass
                
                # Sincronização geral CUDA como último recurso
                torch.cuda.synchronize()
            except Exception as cleanup_error:
                print(f"Erro durante limpeza após falha: {str(cleanup_error)}")
            
            # Retorna resultados parciais
            return [r for r in results if r is not None]
    
    def synchronize_all(self):
        """
        Sincroniza todos os streams no pool.
        """
        if self.device != "cuda":
            return
            
        for stream in self.streams:
            stream.synchronize() 