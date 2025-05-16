import time
import threading
import queue
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, Signal, QCoreApplication

class LoadingProgressDialog(QDialog):
    """
    Diálogo de progresso para carregamento de modelos, exibindo:
    - Mensagem de status
    - Barra de progresso
    - Tempo decorrido
    - Tempo estimado
    - Botão para cancelar (opcional)
    """
    
    # Sinal interno para atualização segura da UI (thread-safe)
    _update_ui_signal = Signal(int, str, float)
    
    def __init__(self, parent=None, cancellable=False):
        super(LoadingProgressDialog, self).__init__(parent)
        self.main_window = parent  # Guarda referência para MainWindow para tradução
        self.setWindowTitle(self.tr_text('Carregando modelo'))
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.resize(450, 180)  # Tamanho aumentado para acomodar mensagens maiores
        self.setMinimumSize(450, 180)
        self.setModal(False)  # Alterado para False para evitar bloqueio da UI
        
        # Inicializa variáveis de tempo e estado
        self.start_time = 0
        self.elapsed_time = 0
        self.is_loading = False
        self.cancellable = cancellable
        self.active_timer = None
        self.last_update_time = 0
        self.progress_history = []  # Histórico de progresso para cálculos mais precisos
        self.last_progress = 0
        
        # Fila para processamento seguro de atualizações de UI
        self.update_queue = queue.Queue()
        self.update_thread_running = False
        self.update_thread = None
        
        # Cria layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)  # Margens aumentadas
        main_layout.setSpacing(10)  # Espaçamento aumentado
        
        # Status label (mostra qual modelo está sendo carregado)
        self.status_label = QLabel(self.tr_text("Preparando para carregar modelo..."))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)  # Permite quebra de texto
        self.status_label.setMinimumHeight(40)  # Altura mínima para 2-3 linhas
        self.status_label.setStyleSheet("font-size: 11pt;")  # Fonte um pouco maior
        main_layout.addWidget(self.status_label)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(25)  # Barra mais alta
        main_layout.addWidget(self.progress_bar)
        
        # Label de tempo decorrido
        self.time_label = QLabel(self.tr_text("Tempo decorrido: 0s"))
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 10pt;")
        main_layout.addWidget(self.time_label)
        
        # Label para tempo estimado (opcional)
        self.estimate_label = QLabel("")
        self.estimate_label.setAlignment(Qt.AlignCenter)
        self.estimate_label.setStyleSheet("font-size: 10pt; color: #555;")
        main_layout.addWidget(self.estimate_label)
        
        # Adiciona botão de cancelar (se habilitado)
        if cancellable:
            button_layout = QHBoxLayout()
            self.cancel_button = QPushButton(self.tr_text("Cancelar"))
            self.cancel_button.clicked.connect(self.reject)
            button_layout.addStretch()
            button_layout.addWidget(self.cancel_button)
            main_layout.addLayout(button_layout)
        
        # Timer para atualizar o tempo decorrido
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.setInterval(100)  # Atualiza a cada 100ms
        
        # Conecta o sinal interno ao atualizador de UI
        self._update_ui_signal.connect(self._perform_ui_update)
        
        # Configura o layout
        self.setLayout(main_layout)
    
    def tr_text(self, text):
        """Função auxiliar para tradução que usa o tr da MainWindow se disponível"""
        if hasattr(self.main_window, 'tr'):
            return self.main_window.tr(text)
        return text
    
    def _start_update_thread(self):
        """Inicia uma thread dedicada para processar atualizações de UI de forma segura"""
        if self.update_thread_running:
            return
            
        self.update_thread_running = True
        self.update_thread = threading.Thread(target=self._process_update_queue)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _process_update_queue(self):
        """Thread de processamento de atualizações da fila para a UI"""
        try:
            while self.update_thread_running:
                try:
                    # Obtém a próxima atualização da fila
                    progress, status_text, timestamp = self.update_queue.get(timeout=0.2)
                    
                    # Calcula o tempo decorrido baseado no timestamp
                    elapsed = timestamp - self.start_time
                    
                    # Envia para a UI via signal (thread-safe)
                    self._update_ui_signal.emit(progress, status_text, elapsed)
                    
                    # Marca como concluído
                    self.update_queue.task_done()
                    
                except queue.Empty:
                    # Sem atualizações por enquanto, apenas continua
                    continue
        except Exception as e:
            print(f"Erro na thread de atualização: {e}")
        finally:
            self.update_thread_running = False
    
    def _perform_ui_update(self, progress, status_text, elapsed_time):
        """Atualiza efetivamente a UI com os valores da fila (chamado na thread principal)"""
        # Atualiza o texto de status se fornecido
        if status_text:
            self.status_label.setText(status_text)
        
        # Atualiza a barra de progresso
        self.progress_bar.setValue(progress)
        
        # Atualiza o tempo decorrido usando o valor fornecido
        self.elapsed_time = elapsed_time
        if self.elapsed_time < 60:
            time_str = f"{self.elapsed_time:.1f}s"
        elif self.elapsed_time < 3600:
            minutes = int(self.elapsed_time // 60)
            seconds = self.elapsed_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(self.elapsed_time // 3600)
            minutes = int((self.elapsed_time % 3600) // 60)
            seconds = self.elapsed_time % 60
            time_str = f"{hours}h {minutes}m {seconds:.0f}s"
        
        # Usar tr_text para tradução
        self.time_label.setText(f"{self.tr_text('Tempo decorrido')}: {time_str}")
        
        # Atualiza o histórico de progresso
        current_time = time.time()
        if progress != self.last_progress:
            self.progress_history.append((current_time, progress))
            # Mantém apenas os últimos 20 pontos de progresso para estimativas
            if len(self.progress_history) > 20:
                self.progress_history = self.progress_history[-20:]
            self.last_progress = progress
            
            # Calcula estimativa de tempo se tiver progresso suficiente
            if progress > 5 and self.elapsed_time > 1.0 and len(self.progress_history) >= 2:
                self._update_time_estimate(progress)
        
        # Processa eventos para manter responsividade
        QCoreApplication.processEvents()
    
    def start_loading(self, model_name):
        """Inicia o processo de carregamento, mostrando o nome do modelo"""
        try:
            self.status_label.setText(self.tr_text('Carregando modelo: {0}').format(model_name))
            self.progress_bar.setValue(0)
            self.start_time = time.time()
            self.elapsed_time = 0
            self.time_label.setText(self.tr_text('Tempo decorrido') + ': 0s')
            self.estimate_label.setText("")
            self.is_loading = True
            self.progress_history = []  # Limpa histórico de progresso
            self.last_progress = 0
            
            # Inicia a thread de processamento de atualizações
            self._start_update_thread()
            
            # Ajusta o tamanho do diálogo de acordo com o conteúdo
            self.adjustSize()
            
            # Para garantir que o timer anterior seja interrompido
            if self.timer.isActive():
                self.timer.stop()
                
            # Inicia um novo timer
            self.timer.start()
            
            # Usa QTimer.singleShot para garantir que a interface seja atualizada
            # antes de mostrar o diálogo
            QTimer.singleShot(10, self.ensure_dialog_shown)
            
            # Processa eventos imediatamente para garantir que a UI seja atualizada
            QCoreApplication.processEvents()
        except Exception as e:
            print(f"Erro ao iniciar diálogo de carregamento: {str(e)}")
    
    def ensure_dialog_shown(self):
        """Garante que o diálogo seja exibido, mesmo em caso de sobrecarga da UI"""
        if not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()
            # Processa eventos imediatamente para garantir visibilidade
            QCoreApplication.processEvents()
    
    def update_progress(self, value, status_text=None):
        """Atualiza o valor da barra de progresso (0-100) e opcionalmente o texto de status"""
        if self.is_loading and 0 <= value <= 100:
            # Adiciona à fila de atualização
            timestamp = time.time()
            self.update_queue.put((value, status_text, timestamp))
            
            # Garante que a thread de processamento esteja rodando
            self._start_update_thread()
            
            # Também atualiza a UI diretamente para garantir responsividade imediata
            # para updates críticos (0%, 100%, ou múltiplos de 10%)
            if value == 0 or value == 100 or value % 10 == 0:
                elapsed = timestamp - self.start_time
                self._update_ui_signal.emit(value, status_text, elapsed)
    
    def _update_time_estimate(self, current_progress):
        """Calcula e atualiza a estimativa de tempo restante"""
        try:
            # Apenas calcula se houver progresso a ser feito
            if current_progress < 100:
                # Obter o primeiro e último ponto do histórico
                first_time, first_progress = self.progress_history[0]
                last_time, last_progress = self.progress_history[-1]
                
                # Calcula a taxa de progresso (% por segundo)
                time_diff = last_time - first_time
                progress_diff = last_progress - first_progress
                
                if time_diff > 0 and progress_diff > 0:
                    # % por segundo
                    progress_rate = progress_diff / time_diff
                    
                    # Tempo restante estimado em segundos
                    remaining_progress = 100 - current_progress
                    estimated_remaining = remaining_progress / progress_rate if progress_rate > 0 else 0
                    
                    # Se a estimativa for muito alta (mais de 2 horas), limita
                    if estimated_remaining > 7200:
                        estimated_remaining = 7200
                    
                    # Formata o tempo restante
                    if estimated_remaining < 60:
                        estimate_str = f"{estimated_remaining:.1f}s"
                    elif estimated_remaining < 3600:
                        minutes = int(estimated_remaining // 60)
                        seconds = estimated_remaining % 60
                        estimate_str = f"{minutes}m {seconds:.0f}s"
                    else:
                        hours = int(estimated_remaining // 3600)
                        minutes = int((estimated_remaining % 3600) // 60)
                        estimate_str = f"{hours}h {minutes}m"
                    
                    # Atualiza o rótulo de estimativa com texto traduzido
                    self.estimate_label.setText(f"{self.tr_text('Tempo estimado restante')}: {estimate_str}")
            else:
                # Limpa a estimativa quando o progresso for 100%
                self.estimate_label.setText("")
        except Exception as e:
            # Ignora erros na estimativa de tempo
            print(f"Erro ao calcular estimativa: {str(e)}")
    
    def update_elapsed_time(self):
        """Atualiza o contador de tempo decorrido com base no timer interno"""
        if self.is_loading:
            try:
                # Calcula o tempo decorrido atual
                current_time = time.time()
                self.elapsed_time = current_time - self.start_time
                
                # Formata o tempo decorrido
                if self.elapsed_time < 60:
                    time_str = f"{self.elapsed_time:.1f}s"
                elif self.elapsed_time < 3600:
                    minutes = int(self.elapsed_time // 60)
                    seconds = self.elapsed_time % 60
                    time_str = f"{minutes}m {seconds:.1f}s"
                else:
                    hours = int(self.elapsed_time // 3600)
                    minutes = int((self.elapsed_time % 3600) // 60)
                    seconds = self.elapsed_time % 60
                    time_str = f"{hours}h {minutes}m {seconds:.0f}s"
                
                # Atualiza o rótulo com texto traduzido
                self.time_label.setText(f"{self.tr_text('Tempo decorrido')}: {time_str}")
                
                # Atualiza a interface com o novo tempo decorrido
                # Usa o mesmo mecanismo seguro de atualização da UI
                self.update_queue.put((self.last_progress, None, current_time))
                
                # Garante que a thread de processamento esteja rodando
                self._start_update_thread()
                
                # Garante que o diálogo continue visível durante o carregamento
                if not self.isVisible():
                    self.ensure_dialog_shown()
            except Exception as e:
                print(f"Erro ao atualizar tempo decorrido: {str(e)}")
    
    def finish_loading(self, success=True):
        """Finaliza o processo de carregamento, mostrando status de sucesso/falha"""
        if self.timer.isActive():
            self.timer.stop()
        
        self.is_loading = False
        
        # Termina a thread de processamento
        if self.update_thread_running:
            self.update_thread_running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
        
        # Limpa a fila de atualizações
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()
                self.update_queue.task_done()
            except queue.Empty:
                break
        
        # Configura a mensagem final baseada no resultado
        if success:
            self.status_label.setText(self.tr_text('Carregamento concluído com sucesso!'))
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(self.tr_text('Falha no carregamento do modelo!'))
            
        # Limpa a estimativa de tempo restante
        self.estimate_label.setText("")
            
        # Mantém o diálogo visível por um curto período antes de fechar
        # Cancela qualquer timer anterior para evitar conflitos
        if self.active_timer:
            try:
                self.killTimer(self.active_timer)
                self.active_timer = None
            except:
                pass
        
        # Processa eventos para atualizar a UI
        QCoreApplication.processEvents()
        
        # Usa um timer único para fechar o diálogo
        QTimer.singleShot(1500, self.safe_hide)
    
    def safe_hide(self):
        """Esconde o diálogo de forma segura"""
        try:
            if self.isVisible():
                self.hide()
                QCoreApplication.processEvents()
        except Exception as e:
            print(f"Erro ao esconder diálogo: {str(e)}")
    
    def closeEvent(self, event):
        """Garante que o timer seja parado quando o diálogo for fechado"""
        if self.timer.isActive():
            self.timer.stop()
        
        # Termina a thread de processamento
        if self.update_thread_running:
            self.update_thread_running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
                
        self.is_loading = False
        super(LoadingProgressDialog, self).closeEvent(event) 