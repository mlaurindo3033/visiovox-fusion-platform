import torch
import gc
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
from contextlib import contextmanager

class MemoryManager:
    """
    Gerenciador centralizado de memória para otimizar o uso de GPU/CPU.
    Fornece:
    - Pré-alocação e pool de buffers
    - Limpeza periódica automática
    - Monitoramento de uso de memória
    """
    
    def __init__(self, device: str = "cuda", cleanup_interval_seconds: int = 300):
        self.device = device
        self.buffer_pools: Dict[str, List[torch.Tensor]] = {}
        self.last_used: Dict[str, float] = {}
        self.buffer_locks: Dict[str, threading.Lock] = {}
        self.model_usage_times: Dict[str, float] = {}
        self.model_load_order: List[str] = []
        self.max_models_in_memory: int = 5
        self.cleanup_interval = cleanup_interval_seconds
        self.allocated_buffers: Set[str] = set()
        self.is_running = False
        self.cleanup_thread = None

        # Configurar logger
        self.logger = logging.getLogger("MemoryManager")
        
        # Iniciar thread de limpeza automática
        if self.device == "cuda" and torch.cuda.is_available():
            self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Inicia thread de limpeza periódica de memória."""
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            return
            
        self.is_running = True
        self.cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, 
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info(f"Thread de limpeza automática de memória iniciada (intervalo: {self.cleanup_interval}s)")
    
    def stop_cleanup_thread(self):
        """Para thread de limpeza periódica."""
        self.is_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
            self.logger.info("Thread de limpeza automática de memória encerrada")
    
    def _periodic_cleanup(self):
        """Executa limpeza periódica de memória em intervalos definidos."""
        while self.is_running:
            time.sleep(self.cleanup_interval)
            if not self.is_running:
                break
                
            # Verificar uso atual de memória
            if self.device == "cuda" and torch.cuda.is_available():
                before_clean = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)
                
                # Limpar buffers não utilizados recentemente
                self._release_unused_buffers(max_age_seconds=self.cleanup_interval*2)
                
                # Executar coleta de lixo e limpar cache CUDA
                gc.collect()
                torch.cuda.empty_cache()
                
                after_clean = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
                
                self.logger.info(f"Limpeza periódica - Memória GPU liberada: {before_clean-after_clean:.2f}MB, "
                              f"Reservada: {reserved_before-reserved_after:.2f}MB")
    
    def _release_unused_buffers(self, max_age_seconds: float = 600):
        """Libera buffers que não foram usados recentemente."""
        current_time = time.time()
        buffers_released = 0
        
        for buffer_key in list(self.last_used.keys()):
            if current_time - self.last_used[buffer_key] > max_age_seconds:
                if buffer_key in self.buffer_pools:
                    with self._get_lock(buffer_key):
                        if buffer_key in self.buffer_pools:  # Verificar novamente dentro do lock
                            num_buffers = len(self.buffer_pools[buffer_key])
                            self.buffer_pools[buffer_key] = []
                            buffers_released += num_buffers
                            del self.buffer_pools[buffer_key]
                        if buffer_key in self.last_used:
                            del self.last_used[buffer_key]
        
        if buffers_released > 0:
            self.logger.info(f"Liberados {buffers_released} buffers não utilizados recentemente")
    
    def _get_lock(self, key: str) -> threading.Lock:
        """Obtém um lock para um determinado buffer_key, criando-o se necessário."""
        if key not in self.buffer_locks:
            self.buffer_locks[key] = threading.Lock()
        return self.buffer_locks[key]
    
    def get_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                   buffer_key: Optional[str] = None) -> torch.Tensor:
        """
        Obtém um buffer pré-alocado do pool ou cria um novo se necessário.
        
        Args:
            shape: Formato do tensor (ex: (1, 3, 256, 256))
            dtype: Tipo do tensor (ex: torch.float32)
            buffer_key: Chave opcional para identificar o tipo de buffer
            
        Returns:
            Tensor pré-alocado com o formato e tipo especificados
        """
        if buffer_key is None:
            # Gerar chave baseada no formato e tipo se não fornecida
            buffer_key = f"{shape}_{dtype}"
        
        # Atualizar timestamp de uso para este tipo de buffer
        self.last_used[buffer_key] = time.time()
        
        with self._get_lock(buffer_key):
            # Verificar se temos buffers disponíveis deste tipo
            if buffer_key in self.buffer_pools and self.buffer_pools[buffer_key]:
                buffer = self.buffer_pools[buffer_key].pop()
                # Se o shape não for exatamente o mesmo, redimensionar
                if buffer.shape != shape:
                    buffer.resize_(shape)
                return buffer
            
            # Criar novo buffer
            if self.device == "cuda" and torch.cuda.is_available():
                buffer = torch.empty(shape, dtype=dtype, device=torch.device("cuda")).contiguous()
            else:
                buffer = torch.empty(shape, dtype=dtype, device=torch.device("cpu")).contiguous()
            
            self.allocated_buffers.add(id(buffer))
            return buffer
    
    def release_buffer(self, buffer: torch.Tensor, buffer_key: Optional[str] = None):
        """
        Devolve um buffer ao pool para reutilização.
        
        Args:
            buffer: Tensor a ser devolvido ao pool
            buffer_key: Chave do tipo de buffer (opcional)
        """
        if buffer is None:
            return
            
        if buffer_key is None:
            buffer_key = f"{buffer.shape}_{buffer.dtype}"
        
        # Verificar se o buffer ainda é válido
        if not isinstance(buffer, torch.Tensor):
            return
            
        buffer_id = id(buffer)
        if buffer_id not in self.allocated_buffers:
            return
            
        with self._get_lock(buffer_key):
            if buffer_key not in self.buffer_pools:
                self.buffer_pools[buffer_key] = []
            self.buffer_pools[buffer_key].append(buffer)
            self.last_used[buffer_key] = time.time()
    
    @contextmanager
    def buffer_context(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                      buffer_key: Optional[str] = None):
        """
        Contexto para uso automático e liberação de buffer.
        
        Exemplo:
            with memory_manager.buffer_context((1, 3, 256, 256), torch.float32) as buffer:
                # Usar o buffer...
                
        Args:
            shape: Formato do tensor
            dtype: Tipo do tensor
            buffer_key: Chave opcional para o tipo de buffer
        """
        buffer = self.get_buffer(shape, dtype, buffer_key)
        try:
            yield buffer
        finally:
            self.release_buffer(buffer, buffer_key)
    
    def register_model_usage(self, model_name: str):
        """
        Registra o uso de um modelo para gerenciamento de lazy loading.
        
        Args:
            model_name: Nome do modelo usado
        """
        self.model_usage_times[model_name] = time.time()
        
        # Manter a ordenação de uso (mais recente para mais antigo)
        if model_name in self.model_load_order:
            self.model_load_order.remove(model_name)
        self.model_load_order.insert(0, model_name)
    
    def should_unload_models(self) -> List[str]:
        """
        Determina quais modelos devem ser descarregados com base no uso.
        
        Returns:
            Lista de nomes de modelos que podem ser descarregados
        """
        # Se temos menos modelos que o limite, não descarregar nenhum
        if len(self.model_load_order) <= self.max_models_in_memory:
            return []
            
        # Retornar modelos menos usados recentemente que excedem o limite
        return self.model_load_order[self.max_models_in_memory:]
    
    def clear_cuda_cache(self, force: bool = False):
        """
        Limpa o cache CUDA para liberar memória.
        
        Args:
            force: Se True, força a limpeza mesmo que não seja necessário
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return
            
        # Verificar se é necessário limpar (se mais de 80% da memória está em uso)
        if force or self.get_gpu_memory_usage() > 0.8:
            before_clean = torch.cuda.memory_allocated() / (1024 ** 2)
            
            # Executar coleta de lixo e limpar cache
            gc.collect()
            torch.cuda.empty_cache()
            
            after_clean = torch.cuda.memory_allocated() / (1024 ** 2)
            self.logger.info(f"Limpeza CUDA - Memória liberada: {before_clean-after_clean:.2f}MB")
    
    def get_gpu_memory_usage(self) -> float:
        """
        Obtém a porcentagem de memória GPU em uso.
        
        Returns:
            Porcentagem de memória utilizada (0.0 a 1.0)
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0
            
        # Obter estatísticas de memória
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return allocated / total
    
    def log_memory_stats(self):
        """Registra estatísticas atuais de memória no log."""
        if self.device != "cuda" or not torch.cuda.is_available():
            self.logger.info(f"Estatísticas de memória - Modo CPU, GPU não disponível")
            return
            
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        
        # Contagem de buffers por tipo
        buffer_counts = {k: len(v) for k, v in self.buffer_pools.items()}
        total_buffers = sum(len(v) for v in self.buffer_pools.values())
        
        self.logger.info(f"Estatísticas de memória - "
                      f"GPU: {allocated:.2f}MB alocado / {reserved:.2f}MB reservado / {total:.2f}MB total, "
                      f"Buffers em pool: {total_buffers}")
    
    def __del__(self):
        """Limpeza quando o gerenciador é destruído."""
        self.stop_cleanup_thread()
        self.clear_cuda_cache(force=True)


# Criar instância global do gerenciador de memória
memory_manager = MemoryManager() 