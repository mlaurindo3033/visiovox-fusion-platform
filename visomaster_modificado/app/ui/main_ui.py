from typing import Dict, Optional # Adicionado Optional para type hints
from pathlib import Path
from functools import partial
import copy
import os
import json

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                               QSlider, QPushButton, QMessageBox, QTabWidget, QCheckBox, 
                               QSpinBox, QLineEdit, QFileDialog, QMainWindow) # QMainWindow adicionado

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import (GraphicsViewEventFilter, VideoSeekSliderEventFilter, 
                                          videoSeekSliderLineEditEventFilter, ListWidgetEventFilter)
from app.ui.widgets import ui_workers
from app.ui.widgets.progress_dialog import LoadingProgressDialog
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.helpers.typing_helper import (FacesParametersTypes, ParametersTypes, ControlTypes, 
                                       MarkerTypes)
# Import movido para o topo
from app.ui.widgets.widget_components import ParameterResetDefaultButton 

ParametersWidgetTypes = Dict[str, widget_components.ToggleButton|widget_components.SelectionBox|widget_components.ParameterDecimalSlider|widget_components.ParameterSlider|widget_components.ParameterText]

CONFIG_FILE = 'config.json'

# Dicionário de traduções CORRIGIDO
TRANSLATIONS = {
    'Português': {
        'Configurações': 'Configurações',
        'Preferências...': 'Preferências...',
        'Geral': 'Geral',
        'Desempenho': 'Desempenho',
        'Cache': 'Cache',
        'Logs': 'Logs',
        'Idioma:': 'Idioma:',
        'Tema:': 'Tema:',
        'Dispositivo de processamento:': 'Dispositivo de processamento:',
        'Número de threads:': 'Número de threads:',
        'Limite de VRAM (MB):': 'Limite de VRAM (MB):',
        'Formato do cache:': 'Formato do cache:',
        'Qualidade (JPEG/WebP):': 'Qualidade (JPEG/WebP):',
        'Limpar cache de frames': 'Limpar cache de frames',
        'Nível de log:': 'Nível de log:',
        'Salvar logs em arquivo': 'Salvar logs em arquivo',
        'OK': 'OK',
        'Cancelar': 'Cancelar',
        'Retroceder Frame': 'Retroceder Frame',
        'Avançar Frame': 'Avançar Frame',
        'Reproduzir': 'Reproduzir',
        'Gravar': 'Gravar',
        'Adicionar Marcador': 'Adicionar Marcador',
        'Remover Marcador': 'Remover Marcador',
        'Próximo Marcador': 'Próximo Marcador',
        'Marcador Anterior': 'Marcador Anterior',
        'Ir para Frame': 'Ir para Frame',
        'Imagens': 'Imagens',
        'Vídeos': 'Vídeos',
        'Webcams': 'Webcams',
        'Uso de VRAM: %p%': 'Uso de VRAM: %p%',
        'Modelo de Segmentação': 'Modelo de Segmentação',
        'Modelo de Detecção': 'Modelo de Detecção',
        'Dispositivo': 'Dispositivo',
        'Limite de VRAM': 'Limite de VRAM',
        'Formato do Cache': 'Formato do Cache',
        'Qualidade do Cache': 'Qualidade do Cache',
        'Threads': 'Threads',
        'Salvar configurações': 'Salvar configurações',
        'Carregar configurações': 'Carregar configurações',
        'Resetar configurações': 'Resetar configurações',
        'Ajuda': 'Ajuda',
        'Escolha o modelo de segmentação facial': 'Escolha o modelo de segmentação facial',
        'Escolha o modelo de detecção facial': 'Escolha o modelo de detecção facial',
        'Selecione o dispositivo de processamento (CPU/GPU)': 'Selecione o dispositivo de processamento (CPU/GPU)',
        'Defina o limite de uso de VRAM para a GPU': 'Defina o limite de uso de VRAM para a GPU',
        'Formato de compressão do cache de frames': 'Formato de compressão do cache de frames',
        'Qualidade da compressão do cache de frames': 'Qualidade da compressão do cache de frames',
        'Número de threads para processamento em lote': 'Número de threads para processamento em lote',
        'Salvar as configurações atuais em um arquivo': 'Salvar as configurações atuais em um arquivo',
        'Carregar configurações de um arquivo': 'Carregar configurações de um arquivo',
        'Restaurar configurações padrão': 'Restaurar configurações padrão',
        'Abrir documentação de ajuda': 'Abrir documentação de ajuda',
        'Red': 'Vermelho',
        'Green': 'Verde',
        'Blue': 'Azul',
        'Blend Amount': 'Quantidade de Mistura',
        'Color Adjustments': 'Ajustes de Cor',
        'Brightness': 'Brilho',
        'Contrast': 'Contraste',
        'Saturation': 'Saturação',
        'Sharpness': 'Nitidez',
        'Hue': 'Matiz',
        'Gamma': 'Gama',
        'Noise': 'Ruído',
        'JPEG Compression': 'Compressão JPEG',
        'Compression': 'Compressão',
        'Final Blend': 'Mistura Final',
        'Final Blend Amount': 'Quantidade de Mistura Final',
        'Overall Mask Blend Amount': 'Quantidade de Mistura da Máscara Geral',
        'General': 'Geral',
        'Providers Priority': 'Prioridade dos Provedores',
        'Benchmark Threads': 'Threads de Benchmark',
        'Método de Segmentação de Rosto': 'Método de Segmentação de Rosto',
        'Modelo SAM 2': 'Modelo SAM 2',
        'Video Settings': 'Configurações de Vídeo',
        'Set Custom Video Playback FPS': 'Definir FPS de Reprodução de Vídeo',
        'Video Playback FPS': 'FPS de Reprodução de Vídeo',
        'Auto Swap': 'Troca Automática',
        'Detectors': 'Detectores',
        'Face Detect Model': 'Modelo de Detecção de Face',
        'Detect Score': 'Pontuação de Detecção',
        'Max No of Faces to Detect': 'Máximo de Faces para Detectar',
        'Auto Rotation': 'Rotação Automática',
        'Manual Rotation': 'Rotação Manual',
        'Rotation Angle': 'Ângulo de Rotação',
        'Enable Landmark Detection': 'Ativar Detecção de Pontos',
        'Landmark Detect Model': 'Modelo de Detecção de Pontos',
        'Landmark Detect Score': 'Pontuação de Detecção de Pontos',
        'Face Landmarks Correction': 'Correção Faciais',
        'Face Similarity:': 'Similaridade Facial',
        'Blur Amount:': 'Quantidade de Desfoque',
        'Detect From Points': 'Detectar a partir de Pontos',
        'Show Landmarks': 'Mostrar Pontos',
        'Show Bounding Boxes': 'Mostrar Caixas Delimitadoras',
        'DFM Settings': 'Configurações DFM',
        'Maximum DFM Models to use': 'Máximo de Modelos DFM para usar',
        'Frame Enhancer': 'Aprimorador de Quadros',
        'Enable Frame Enhancer': 'Ativar Aprimorador de Quadros',
        'Frame Enhancer Type': 'Tipo de Aprimorador de Quadros',
        'Blend': 'Mistura',
        'Webcam Settings': 'Configurações da Webcam',
        'Webcam Max No': 'Máximo de Webcams',
        'Webcam Backend': 'Backend da Webcam',
        'Webcam Resolution': 'Resolução da Webcam',
        'Webcam FPS': 'FPS da Webcam',
        'Virtual Camera': 'Câmera Virtual',
        'Send Frames to Virtual Camera': 'Enviar Quadros para Câmera Virtual',
        'Virtual Camera Backend': 'Backend da Câmera Virtual',
        'Face Recognition': 'Reconhecimento Facial',
        'Recognition Model': 'Modelo de Reconhecimento',
        'Swapping Similarity Type': 'Tipo de Similaridade de Troca',
        'Embedding Merge Method': 'Método de Mesclagem de Embeddings',
        'Media Selection': 'Seleção de Mídia',
        'Target Media Include Subfolders': 'Mídia Alvo Inclui Subpastas',
        'Input Faces Include Subfolders': 'Faces de Entrada Incluem Subpastas',
        'Enable Face Pose/Expression Editor': 'Ativar Editor de Pose/Expressão Facial',
        'Face Editor Type': 'Tipo de Editor Facial',
        'Eyes Close <--> Open Ratio': 'Proporção Olhos Fechados <--> Abertos',
        'Lips Close <--> Open Ratio': 'Proporção Lábios Fechados <--> Abertos',
        'Head Pitch': 'Inclinação da Cabeça',
        'Head Yaw': 'Giro da Cabeça',
        'Head Roll': 'Rolar da Cabeça',
        'X-Axis Movement': 'Movimento no Eixo X',
        'Y-Axis Movement': 'Movimento no Eixo Y',
        'Z-Axis Movement': 'Movimento no Eixo Z',
        'Mouth Pouting': 'Bico com a Boca',
        'Mouth Pursing': 'Boca Cerrada',
        'Mouth Grin': 'Sorriso Largo',
        'Lips Close <--> Open Value': 'Valor Lábios Fechados <--> Abertos',
        'Mouth Smile': 'Sorriso',
        'Eye Wink': 'Piscar de Olho',
        'EyeBrows Direction': 'Direção das Sobrancelhas',
        'EyeGaze Horizontal': 'Olhar Horizontal',
        'EyeGaze Vertical': 'Olhar Vertical',
        'Face Makeup': 'Maquiagem Facial',
        'Hair Makeup': 'Maquiagem de Cabelo',
        'Lips Makeup': 'Maquiagem de Lábios',
        'EyeBrows Makeup': 'Maquiagem de Sobrancelhas',
        'Enable Face Restorer': 'Ativar Restaurador Facial',
        'Restorer Type': 'Tipo de Restaurador',
        'Alignment': 'Alinhamento',
        'Fidelity Weight': 'Peso de Fidelidade',
        'Enable Face Restorer 2': 'Ativar Restaurador Facial 2',
        'Enable Face Expression Restorer': 'Ativar Restaurador de Expressão Facial',
        'Crop Scale': 'Escala de Recorte',
        'VY Ratio': 'Proporção VY',
        'Expression Friendly Factor': 'Fator Amigável de Expressão',
        'Animation Region': 'Região de Animação',
        'Normalize Lips': 'Normalizar Lábios',
        'Normalize Lips Threshold': 'Limite de Normalização dos Lábios',
        'Retargeting Eyes': 'Redirecionamento de Olhos',
        'Retargeting Eyes Multiplier': 'Multiplicador de Redirecionamento de Olhos',
        'Retargeting Lips': 'Redirecionamento de Lábios',
        'Retargeting Lips Multiplier': 'Multiplicador de Redirecionamento de Lábios',
        'Swapper Model': 'Modelo Trocador',
        'Swapper Resolution': 'Resolução do Trocador',
        'DFM Model': 'Modelo DFM',
        'AMP Morph Factor': 'Fator de Morfologia AMP',
        'RCT Color Transfer': 'Transferência de Cor RCT',
        'Face Adjustments': 'Ajustes Faciais',
        'Keypoints X-Axis': 'Pontos-Chave Eixo X',
        'Keypoints Y-Axis': 'Pontos-Chave Eixo Y',
        'Keypoints Scale': 'Escala dos Pontos-Chave',
        'Face Scale Amount': 'Quantidade de Escala da Face',
        '5 - Keypoints Adjustments': 'Ajustes de 5 Pontos-Chave',
        'Left Eye:   X': 'Olho Esquerdo:   X',
        'Left Eye:   Y': 'Olho Esquerdo:   Y',
        'Right Eye:   X': 'Olho Direito:   X',
        'Right Eye:   Y': 'Olho Direito:   Y',
        'Nose:   X': 'Nariz:   X',
        'Nose:   Y': 'Nariz:   Y',
        'Left Mouth:   X': 'Boca Esquerda:   X',
        'Left Mouth:   Y': 'Boca Esquerda:   Y',
        'Right Mouth:   X': 'Boca Direita:   X',
        'Right Mouth:   Y': 'Boca Direita:   Y',
        'Similarity Threshold': 'Limite de Similaridade',
        'Strength': 'Força',
        'Amount': 'Quantidade',
        'Face Likeness': 'Semelhança Facial',
        'Differencing': 'Diferenciação',
        'Bottom Border': 'Borda Inferior',
        'Left Border': 'Borda Esquerda',
        'Right Border': 'Borda Direita',
        'Top Border': 'Borda Superior',
        'Border Blur': 'Desfoque da Borda',
        'Occlusion Mask': 'Máscara de Oclusão',
        'Size': 'Tamanho',
        'DFL XSeg Mask': 'Máscara DFL XSeg',
        'Occluder/DFL XSeg Blur': 'Desfoque Oclusor/DFL XSeg',
        'Text Masking': 'Mascaramento de Texto',
        'Text Masking Entry': 'Entrada de Mascaramento de Texto',
        'Face Parser Mask': 'Máscara do Analisador Facial',
        'Background': 'Fundo',
        'Face': 'Rosto',
        'Left Eyebrow': 'Sobrancelha Esquerda',
        'Right Eyebrow': 'Sobrancelha Direita',
        'Left Eye': 'Olho Esquerdo',
        'Right Eye': 'Olho Direito',
        'EyeGlasses': 'Óculos',
        'Nose': 'Nariz',
        'Mouth': 'Boca',
        'Upper Lip': 'Lábio Superior',
        'Lower Lip': 'Lábio Inferior',
        'Neck': 'Pescoço',
        'Hair': 'Cabelo',
        'Background Blur': 'Desfoque do Fundo',
        'Face Blur': 'Desfoque do Rosto',
        'Lips Makeup': 'Maquiagem dos Lábios',
        'Restore Eyes': 'Restaurar Olhos',
        'Eyes Blend Amount': 'Quantidade de Mistura dos Olhos',
        'Eyes Size Factor': 'Fator de Tamanho dos Olhos',
        'Eyes Feather Blend': 'Mistura Suave dos Olhos',
        'X Eyes Radius Factor': 'Fator de Raio X dos Olhos',
        'Y Eyes Radius Factor': 'Fator de Raio Y dos Olhos',
        'X Eyes Offset': 'Deslocamento X dos Olhos',
        'Y Eyes Offset': 'Deslocamento Y dos Olhos',
        'Eyes Spacing Offset': 'Deslocamento de Espaçamento dos Olhos',
        'Restore Mouth': 'Restaurar Boca',
        'Mouth Blend Amount': 'Quantidade de Mistura da Boca',
        'Mouth Size Factor': 'Fator de Tamanho da Boca',
        'Mouth Feather Blend': 'Mistura Suave da Boca',
        'X Mouth Radius Factor': 'Fator de Raio X da Boca',
        'Y Mouth Radius Factor': 'Fator de Raio Y da Boca',
        'X Mouth Offset': 'Deslocamento X da Boca',
        'Y Mouth Offset': 'Deslocamento Y da Boca',
        'Eyes/Mouth Blur': 'Desfoque Olhos/Boca',
        'Face Color Correction': 'Correção de Cor do Rosto',
        'AutoColor Transfer': 'Transferência Automática de Cor',
        'Transfer Type': 'Tipo de Transferência',
        'Color Adjustments': 'Ajustes de Cor',
        'JPEG Compression': 'Compressão JPEG',
        'Blend Adjustments': 'Ajustes de Mistura',
        'Final Blend': 'Mistura Final',
        'Other Options': 'Outras Opções',
        'Face Editor': 'Editor Facial',
        'Common': 'Comum',
        'Face Restorer': 'Restaurador Facial',
        'Detector Settings': 'Configurações do Detector',
        'File': 'Arquivo',
        'Edit': 'Editar',
        'View': 'Visualizar',
        'Target Videos and Inputs': 'Vídeos e Entradas Alvo',
        'Target Videos/Images': 'Vídeos/Imagens Alvo',
        'Select Videos/Images Path': 'Selecionar Caminho de Vídeos/Imagens',
        'Search Videos/Images': 'Buscar Vídeos/Imagens',
        'Drop Files or Click here to Select a Folder': 'Arraste arquivos ou clique para selecionar pasta',
        'Input Faces': 'Faces de Entrada',
        'Select Face Images Path': 'Selecionar Caminho das Faces',
        'Search Faces': 'Buscar Faces',
        'Save Image': 'Salvar Imagem',
        'Find Faces': 'Encontrar Faces',
        'Clear Faces': 'Limpar Faces',
        'Swap Faces': 'Trocar Faces',
        'Edit Faces': 'Editar Faces',
        'Search Embeddings': 'Buscar Embeddings',
        'Media Panel': 'Painel de Mídia',
        'Faces Panel': 'Painel de Faces',
        'Parameters Panel': 'Painel de Parâmetros',
        'View Face Compare': 'Ver Comparação',
        'View Face Mask': 'Ver Máscara',
        'Control Options': 'Opções de Controle',
        'Clear VRAM': 'Limpar VRAM',
        'Face Swap': 'Troca de Rosto',
        'Swapper': 'Trocador',
        'Keypoints Adjustments': 'Ajustes dos Pontos-chave',
        'Left Eye: X': 'Olho Esquerdo: X',
        'Left Eye: Y': 'Olho Esquerdo: Y',
        'Right Eye: X': 'Olho Direito: X',
        'Right Eye: Y': 'Olho Direito: Y',
        'Nose: X': 'Nariz: X',
        'Nose: Y': 'Nariz: Y',
        'Left Mouth: X': 'Boca Esquerda: X',
        'Left Mouth: Y': 'Boca Esquerda: Y',
        'Right Mouth: X': 'Boca Direita: X',
        'Right Mouth: Y': 'Boca Direita: Y',
        'Face Mask': 'Máscara Facial',
        # Strings do LoadingProgressDialog
        'Carregando modelo': 'Carregando modelo',
        'Preparando para carregar modelo...': 'Preparando para carregar modelo...',
        'Tempo decorrido': 'Tempo decorrido',
        'Tempo estimado restante': 'Tempo estimado restante',
        'Carregamento concluído com sucesso!': 'Carregamento concluído com sucesso!',
        'Falha no carregamento do modelo!': 'Falha no carregamento do modelo!',
        'Carregando modelo: {0}': 'Carregando modelo: {0}',
        'Cancelar': 'Cancelar',
    },
    'English': {
        'Configurações': 'Settings',
        'Preferências...': 'Preferences...',
        'Geral': 'General',
        'Desempenho': 'Performance',
        'Cache': 'Cache',
        'Logs': 'Logs',
        'Idioma:': 'Language:',
        'Tema:': 'Theme:',
        'Dispositivo de processamento:': 'Processing device:',
        'Número de threads:': 'Number of threads:',
        'Limite de VRAM (MB):': 'VRAM limit (MB):',
        'Formato do cache:': 'Cache format:',
        'Qualidade (JPEG/WebP):': 'Quality (JPEG/WebP):',
        'Limpar cache de frames': 'Clear frame cache',
        'Nível de log:': 'Log level:',
        'Salvar logs em arquivo': 'Save logs to file',
        'OK': 'OK',
        'Cancelar': 'Cancel',
        'Retroceder Frame': 'Rewind Frame',
        'Avançar Frame': 'Advance Frame',
        'Reproduzir': 'Play',
        'Gravar': 'Record',
        'Adicionar Marcador': 'Add Marker',
        'Remover Marcador': 'Remove Marker',
        'Próximo Marcador': 'Next Marker',
        'Marcador Anterior': 'Previous Marker',
        'Ir para Frame': 'Go to Frame',
        'Imagens': 'Images',
        'Vídeos': 'Videos',
        'Webcams': 'Webcams',
        'Uso de VRAM: %p%': 'VRAM Usage: %p%',
        'Modelo de Segmentação': 'Segmentation Model',
        'Modelo de Detecção': 'Detection Model',
        'Dispositivo': 'Device',
        'Limite de VRAM': 'VRAM Limit',
        'Formato do Cache': 'Cache Format',
        'Qualidade do Cache': 'Cache Quality',
        'Threads': 'Threads',
        'Salvar configurações': 'Save Settings',
        'Carregar configurações': 'Load Settings',
        'Resetar configurações': 'Reset Settings',
        'Ajuda': 'Help',
        'Escolha o modelo de segmentação facial': 'Choose the face segmentation model',
        'Escolha o modelo de detecção facial': 'Choose the face detection model',
        'Selecione o dispositivo de processamento (CPU/GPU)': 'Select the processing device (CPU/GPU)',
        'Defina o limite de uso de VRAM para a GPU': 'Set the VRAM usage limit for the GPU',
        'Formato de compressão do cache de frames': 'Frame cache compression format',
        'Qualidade da compressão do cache de frames': 'Frame cache compression quality',
        'Número de threads para processamento em lote': 'Number of threads for batch processing',
        'Salvar as configurações atuais em um arquivo': 'Save current settings to a file',
        'Carregar configurações de um arquivo': 'Load settings from a file',
        'Restaurar configurações padrão': 'Restore default settings',
        'Abrir documentação de ajuda': 'Open help documentation',
        'Red': 'Red',
        'Green': 'Green',
        'Blue': 'Blue',
        'Blend Amount': 'Blend Amount',
        'Color Adjustments': 'Color Adjustments',
        'Brightness': 'Brightness',
        'Contrast': 'Contrast',
        'Saturation': 'Saturation',
        'Sharpness': 'Sharpness',
        'Hue': 'Hue',
        'Gamma': 'Gamma',
        'Noise': 'Noise',
        'JPEG Compression': 'JPEG Compression',
        'Compression': 'Compression',
        'Final Blend': 'Final Blend',
        'Final Blend Amount': 'Final Blend Amount',
        'Overall Mask Blend Amount': 'Overall Mask Blend Amount',
        'General': 'General',
        'Providers Priority': 'Providers Priority',
        'Benchmark Threads': 'Benchmark Threads',
        'Método de Segmentação de Rosto': 'Face Segmentation Method',
        'Modelo SAM 2': 'SAM Model 2',
        'Video Settings': 'Video Settings',
        'Set Custom Video Playback FPS': 'Set Custom Video Playback FPS',
        'Video Playback FPS': 'Video Playback FPS',
        'Auto Swap': 'Auto Swap',
        'Detectors': 'Detectors',
        'Face Detect Model': 'Face Detect Model',
        'Detect Score': 'Detect Score',
        'Max No of Faces to Detect': 'Max No of Faces to Detect',
        'Auto Rotation': 'Auto Rotation',
        'Manual Rotation': 'Manual Rotation',
        'Rotation Angle': 'Rotation Angle',
        'Enable Landmark Detection': 'Enable Landmark Detection',
        'Landmark Detect Model': 'Landmark Detect Model',
        'Landmark Detect Score': 'Landmark Detect Score',
        'Detect From Points': 'Detect From Points',
        'Show Landmarks': 'Show Landmarks',
        'Show Bounding Boxes': 'Show Bounding Boxes',
        'DFM Settings': 'DFM Settings',
        'Maximum DFM Models to use': 'Maximum DFM Models to use',
        'Frame Enhancer': 'Frame Enhancer',
        'Enable Frame Enhancer': 'Enable Frame Enhancer',
        'Frame Enhancer Type': 'Frame Enhancer Type',
        'Blend': 'Blend',
        'Webcam Settings': 'Webcam Settings',
        'Webcam Max No': 'Webcam Max No',
        'Webcam Backend': 'Webcam Backend',
        'Webcam Resolution': 'Webcam Resolution',
        'Webcam FPS': 'Webcam FPS',
        'Virtual Camera': 'Virtual Camera',
        'Send Frames to Virtual Camera': 'Send Frames to Virtual Camera',
        'Virtual Camera Backend': 'Virtual Camera Backend',
        'Face Recognition': 'Face Recognition',
        'Recognition Model': 'Recognition Model',
        'Swapping Similarity Type': 'Swapping Similarity Type',
        'Embedding Merge Method': 'Embedding Merge Method',
        'Media Selection': 'Media Selection',
        'Target Media Include Subfolders': 'Target Media Include Subfolders',
        'Input Faces Include Subfolders': 'Input Faces Include Subfolders',
        'Enable Face Pose/Expression Editor': 'Enable Face Pose/Expression Editor',
        'Face Editor Type': 'Face Editor Type',
        'Eyes Close <--> Open Ratio': 'Eyes Close <--> Open Ratio',
        'Lips Close <--> Open Ratio': 'Lips Close <--> Open Ratio',
        'Head Pitch': 'Head Pitch',
        'Head Yaw': 'Head Yaw',
        'Head Roll': 'Head Roll',
        'X-Axis Movement': 'X-Axis Movement',
        'Y-Axis Movement': 'Y-Axis Movement',
        'Z-Axis Movement': 'Z-Axis Movement',
        'Mouth Pouting': 'Mouth Pouting',
        'Mouth Pursing': 'Mouth Pursing',
        'Mouth Grin': 'Mouth Grin',
        'Lips Close <--> Open Value': 'Lips Close <--> Open Value',
        'Mouth Smile': 'Mouth Smile',
        'Eye Wink': 'Eye Wink',
        'EyeBrows Direction': 'EyeBrows Direction',
        'EyeGaze Horizontal': 'EyeGaze Horizontal',
        'EyeGaze Vertical': 'EyeGaze Vertical',
        'Face Makeup': 'Face Makeup',
        'Hair Makeup': 'Hair Makeup',
        'Lips Makeup': 'Lips Makeup',
        'EyeBrows Makeup': 'EyeBrows Makeup',
        'Enable Face Restorer': 'Enable Face Restorer',
        'Restorer Type': 'Restorer Type',
        'Alignment': 'Alignment',
        'Fidelity Weight': 'Fidelity Weight',
        'Enable Face Restorer 2': 'Enable Face Restorer 2',
        'Enable Face Expression Restorer': 'Enable Face Expression Restorer',
        'Crop Scale': 'Crop Scale',
        'VY Ratio': 'VY Ratio',
        'Expression Friendly Factor': 'Expression Friendly Factor',
        'Animation Region': 'Animation Region',
        'Normalize Lips': 'Normalize Lips',
        'Normalize Lips Threshold': 'Normalize Lips Threshold',
        'Retargeting Eyes': 'Retargeting Eyes',
        'Retargeting Eyes Multiplier': 'Retargeting Eyes Multiplier',
        'Retargeting Lips': 'Retargeting Lips',
        'Retargeting Lips Multiplier': 'Retargeting Lips Multiplier',
        'Swapper Model': 'Swapper Model',
        'Swapper Resolution': 'Swapper Resolution',
        'DFM Model': 'DFM Model',
        'AMP Morph Factor': 'AMP Morph Factor',
        'RCT Color Transfer': 'RCT Color Transfer',
        'Face Adjustments': 'Face Adjustments',
        'Keypoints X-Axis': 'Keypoints X-Axis',
        'Keypoints Y-Axis': 'Keypoints Y-Axis',
        'Keypoints Scale': 'Keypoints Scale',
        'Face Scale Amount': 'Face Scale Amount',
        '5 - Keypoints Adjustments': '5 - Keypoints Adjustments',
        'Left Eye:   X': 'Left Eye:   X',
        'Left Eye:   Y': 'Left Eye:   Y',
        'Right Eye:   X': 'Right Eye:   X',
        'Right Eye:   Y': 'Right Eye:   Y',
        'Nose:   X': 'Nose:   X',
        'Nose:   Y': 'Nose:   Y',
        'Left Mouth:   X': 'Left Mouth:   X',
        'Left Mouth:   Y': 'Left Mouth:   Y',
        'Right Mouth:   X': 'Right Mouth:   X',
        'Right Mouth:   Y': 'Right Mouth:   Y',
        'Similarity Threshold': 'Similarity Threshold',
        'Strength': 'Strength',
        'Amount': 'Amount',
        'Face Likeness': 'Face Likeness',
        'Differencing': 'Differencing',
        'Bottom Border': 'Bottom Border',
        'Left Border': 'Left Border',
        'Right Border': 'Right Border',
        'Top Border': 'Top Border',
        'Border Blur': 'Border Blur',
        'Occlusion Mask': 'Occlusion Mask',
        'Size': 'Size',
        'DFL XSeg Mask': 'DFL XSeg Mask',
        'Occluder/DFL XSeg Blur': 'Occluder/DFL XSeg Blur',
        'Text Masking': 'Text Masking',
        'Text Masking Entry': 'Text Masking Entry',
        'Face Parser Mask': 'Face Parser Mask',
        'Background': 'Background',
        'Face': 'Face',
        'Left Eyebrow': 'Left Eyebrow',
        'Right Eyebrow': 'Right Eyebrow',
        'Left Eye': 'Left Eye',
        'Right Eye': 'Right Eye',
        'EyeGlasses': 'EyeGlasses',
        'Nose': 'Nose',
        'Mouth': 'Mouth',
        'Upper Lip': 'Upper Lip',
        'Lower Lip': 'Lower Lip',
        'Neck': 'Neck',
        'Hair': 'Hair',
        'Background Blur': 'Background Blur',
        'Face Blur': 'Face Blur',
        'Lips Makeup': 'Lips Makeup',
        'Restore Eyes': 'Restore Eyes',
        'Eyes Blend Amount': 'Eyes Blend Amount',
        'Eyes Size Factor': 'Eyes Size Factor',
        'Eyes Feather Blend': 'Eyes Feather Blend',
        'X Eyes Radius Factor': 'X Eyes Radius Factor',
        'Y Eyes Radius Factor': 'Y Eyes Radius Factor',
        'X Eyes Offset': 'X Eyes Offset',
        'Y Eyes Offset': 'Y Eyes Offset',
        'Eyes Spacing Offset': 'Eyes Spacing Offset',
        'Restore Mouth': 'Restore Mouth',
        'Mouth Blend Amount': 'Mouth Blend Amount',
        'Mouth Size Factor': 'Mouth Size Factor',
        'Mouth Feather Blend': 'Mouth Feather Blend',
        'X Mouth Radius Factor': 'X Mouth Radius Factor',
        'Y Mouth Radius Factor': 'Y Mouth Radius Factor',
        'X Mouth Offset': 'X Mouth Offset',
        'Y Mouth Offset': 'Y Mouth Offset',
        'Eyes/Mouth Blur': 'Eyes/Mouth Blur',
        'Face Color Correction': 'Face Color Correction',
        'AutoColor Transfer': 'AutoColor Transfer',
        'Transfer Type': 'Transfer Type',
        'Color Adjustments': 'Color Adjustments',
        'JPEG Compression': 'JPEG Compression',
        'Blend Adjustments': 'Blend Adjustments',
        'Final Blend': 'Final Blend',
        'Other Options': 'Other Options',
        'Overall Mask Blend Amount': 'Overall Mask Blend Amount',
        'Face Editor': 'Face Editor',
        'Common': 'Common',
        'Face Restorer': 'Face Restorer',
        'Detector Settings': 'Detector Settings',
        'File': 'File',
        'Edit': 'Edit',
        'View': 'View',
        'Target Videos and Inputs': 'Target Videos and Inputs',
        'Target Videos/Images': 'Target Videos/Images',
        'Select Videos/Images Path': 'Select Videos/Images Path',
        'Search Videos/Images': 'Search Videos/Images',
        'Drop Files or Click here to Select a Folder': 'Drop Files or Click here to Select a Folder',
        'Input Faces': 'Input Faces',
        'Select Face Images Path': 'Select Face Images Path',
        'Search Faces': 'Search Faces',
        'Save Image': 'Save Image',
        'Find Faces': 'Find Faces',
        'Clear Faces': 'Clear Faces',
        'Swap Faces': 'Swap Faces',
        'Edit Faces': 'Edit Faces',
        'Search Embeddings': 'Search Embeddings',
        'Media Panel': 'Media Panel',
        'Faces Panel': 'Faces Panel',
        'Parameters Panel': 'Parameters Panel',
        'View Face Compare': 'View Face Compare',
        'View Face Mask': 'View Face Mask',
        'Control Options': 'Control Options',
        'Clear VRAM': 'Clear VRAM',
        'Face Swap': 'Face Swap',
        'Swapper': 'Swapper',
        'Keypoints Adjustments': 'Keypoints Adjustments',
        'Left Eye: X': 'Left Eye: X',
        'Left Eye: Y': 'Left Eye: Y',
        'Right Eye: X': 'Right Eye: X',
        'Right Eye: Y': 'Right Eye: Y',
        'Nose: X': 'Nose: X',
        'Nose: Y': 'Nose: Y',
        'Left Mouth: X': 'Left Mouth: X',
        'Left Mouth: Y': 'Left Mouth: Y',
        'Right Mouth: X': 'Right Mouth: X',
        'Right Mouth: Y': 'Right Mouth: Y',
        'Face Mask': 'Face Mask',
        # LoadingProgressDialog strings
        'Carregando modelo': 'Loading model',
        'Preparando para carregar modelo...': 'Preparing to load model...',
        'Tempo decorrido': 'Elapsed time',
        'Tempo estimado restante': 'Estimated time remaining',
        'Carregamento concluído com sucesso!': 'Model loaded successfully!',
        'Falha no carregamento do modelo!': 'Failed to load model!',
        'Carregando modelo: {0}': 'Loading model: {0}',
        'Cancelar': 'Cancel',
    },
    'Русский': {
        'Configurações': 'Настройки',
        'Preferências...': 'Предпочтения...',
        'Geral': 'Общие',
        'Desempenho': 'Производительность',
        'Cache': 'Кэш',
        'Logs': 'Логи',
        'Idioma:': 'Язык:',
        'Tema:': 'Тема:',
        'Dispositivo de processamento:': 'Устройство обработки:',
        'Número de threads:': 'Количество потоков:',
        'Limite de VRAM (MB):': 'Лимит VRAM (МБ):',
        'Formato do cache:': 'Формат кэша:', # CORRIGIDO
        'Qualidade (JPEG/WebP):': 'Качество (JPEG/WebP):',
        'Limpar cache de frames': 'Очистить кэш кадров',
        'Nível de log:': 'Уровень логирования:',
        'Salvar logs em arquivo': 'Сохранять логи в файл',
        'OK': 'ОК',
        'Cancelar': 'Отмена',
        'Retroceder Frame': 'Назад кадр',
        'Avançar Frame': 'Вперёд кадр',
        'Reproduzir': 'Воспроизвести',
        'Gravar': 'Запись',
        'Adicionar Marcador': 'Добавить маркер',
        'Remover Marcador': 'Удалить маркер',
        'Próximo Marcador': 'Следующий маркер',
        'Marcador Anterior': 'Предыдущий маркер',
        'Ir para Frame': 'Перейти к кадру',
        'Imagens': 'Изображения',
        'Vídeos': 'Видео',
        'Webcams': 'Веб-камеры',
        'Uso de VRAM: %p%': 'Использование VRAM: %p%',
        'Modelo de Segmentação': 'Модель сегментации',
        'Modelo de Detecção': 'Модель обнаружения',
        'Dispositivo': 'Устройство',
        'Limite de VRAM': 'Лимит VRAM',
        'Formato до Cache': 'Качество кэша', # CORRIGIDO
        'Threads': 'Потоки',
        'Salvar configurações': 'Сохранить настройки',
        'Carregar configurações': 'Загрузить настройки',
        'Resetar configurações': 'Сбросить настройки',
        'Ajuda': 'Помощь',
        'Escolha o modelo de segmentação facial': 'Выберите модель сегментации лица',
        'Escolha o modelo de detecção facial': 'Выберите модель обнаружения лица',
        'Selecione o dispositivo de processamento (CPU/GPU)': 'Выберите устройство обработки (CPU/GPU)',
        'Defina o limite de uso de VRAM для a GPU': 'Установите лимит использования VRAM для GPU', # CORRIGIDO
        'Formato de compressão до Cache': 'Качество сжатия кэша кадров', # CORRIGIDO
        'Número de threads для processamento em lote': 'Количество потоков для пакетной обработки', # CORRIGIDO
        'Salvar as configurações atuais em um arquivo': 'Сохранить текущие настройки в файл',
        'Carregar configurações de um arquivo': 'Загрузить настройки из файла',
        'Restaurar configurações padrão': 'Восстановить настройки по умолчанию',
        'Abrir documentação de ajuda': 'Открыть документацию', # CORRIGIDO
        'Red': 'Красный',
        'Green': 'Зелёный',
        'Blue': 'Синий',
        'Blend Amount': 'Степень смешивания',
        'Face Landmarks Correction': 'Коррекция ориентиров лица',
        'Face Similarity:': 'Сходство лиц',
        'Blur Amount:': 'Степень размытия',
        'Color Adjustments': 'Коррекция цвета',
        'Brightness': 'Яркость',
        'Contrast': 'Контраст',
        'Saturation': 'Насыщенность',
        'Sharpness': 'Резкость',
        'Hue': 'Оттенок',
        'Gamma': 'Гамма',
        'Noise': 'Шум',
        'JPEG Compression': 'JPEG сжатие',
        'Compression': 'Сжатие',
        'Final Blend': 'Финальное смешивание',
        'Final Blend Amount': 'Степень финального смешивания',
        'Overall Mask Blend Amount': 'Общее смешивание маски',
        'General': 'Общие',
        'Providers Priority': 'Приоритет провайдеров',
        'Benchmark Threads': 'Потоки тестирования',
        'Método de Segmentação de Rosto': 'Метод сегментации лица',
        'Modelo SAM 2': 'Модель SAM 2',
        'Video Settings': 'Настройки видео',
        'Set Custom Video Playback FPS': 'Установить FPS воспроизведения видео',
        'Video Playback FPS': 'FPS воспроизведения видео',
        'Auto Swap': 'Автоматическая замена',
        'Detectors': 'Детекторы',
        'Face Detect Model': 'Модель обнаружения лица',
        'Detect Score': 'Оценка обнаружения',
        'Max No of Faces to Detect': 'Максимум лиц для обнаружения',
        'Auto Rotation': 'Автоматическое вращение',
        'Manual Rotation': 'Ручное вращение',
        'Rotation Angle': 'Угол вращения',
        'Enable Landmark Detection': 'Включить обнаружение точек',
        'Landmark Detect Model': 'Модель обнаружения точек',
        'Landmark Detect Score': 'Оценка обнаружения точек',
        'Detect From Points': 'Обнаружить по точкам',
        'Show Landmarks': 'Показать точки',
        'Show Bounding Boxes': 'Показать ограничивающие рамки',
        'DFM Settings': 'Настройки DFM',
        'Maximum DFM Models to use': 'Максимум моделей DFM',
        'Frame Enhancer': 'Улучшение кадров',
        'Enable Frame Enhancer': 'Включить улучшение кадров',
        'Frame Enhancer Type': 'Тип улучшения кадров',
        'Blend': 'Смешивание',
        'Webcam Settings': 'Настройки веб-камеры',
        'Webcam Max No': 'Максимум веб-камер',
        'Webcam Backend': 'Бэкенд веб-камеры',
        'Webcam Resolution': 'Разрешение веб-камеры',
        'Webcam FPS': 'FPS веб-камеры',
        'Virtual Camera': 'Виртуальная камера',
        'Send Frames to Virtual Camera': 'Отправить кадры в виртуальную камеру',
        'Virtual Camera Backend': 'Бэкенд виртуальной камеры',
        'Face Recognition': 'Распознавание лиц',
        'Recognition Model': 'Модель распознавания',
        'Swapping Similarity Type': 'Тип сходства при замене',
        'Embedding Merge Method': 'Метод объединения эмбеддингов',
        'Media Selection': 'Выбор медиа',
        'Target Media Include Subfolders': 'Медиа-цели включают подпапки',
        'Input Faces Include Subfolders': 'Входные лица включают подпапки',
        'Enable Face Pose/Expression Editor': 'Включить редактор поз/выражений лица',
        'Face Editor Type': 'Тип редактора лица',
        'Eyes Close <--> Open Ratio': 'Соотношение закрытых/открытых глаз',
        'Lips Close <--> Open Ratio': 'Соотношение закрытых/открытых губ',
        'Head Pitch': 'Наклон головы',
        'Head Yaw': 'Поворот головы',
        'Head Roll': 'Крен головы',
        'X-Axis Movement': 'Движение по оси X',
        'Y-Axis Movement': 'Движение по оси Y',
        'Z-Axis Movement': 'Движение по оси Z',
        'Mouth Pouting': 'Надутые губы',
        'Mouth Pursing': 'Сжатые губы',
        'Mouth Grin': 'Широкая улыбка',
        'Lips Close <--> Open Value': 'Значение закрытых/открытых губ',
        'Mouth Smile': 'Улыбка',
        'Eye Wink': 'Подмигивание',
        'EyeBrows Direction': 'Направление бровей',
        'EyeGaze Horizontal': 'Горизонтальный взгляд',
        'EyeGaze Vertical': 'Вертикальный взгляд',
        'Face Makeup': 'Макияж лица',
        'Hair Makeup': 'Макияж волос',
        'Lips Makeup': 'Макияж губ',
        'EyeBrows Makeup': 'Макияж бровей',
        'Enable Face Restorer': 'Включить Восстановление Лица',
        'Restorer Type': 'Тип Восстановителя',
        'Alignment': 'Выравнивание',
        'Fidelity Weight': 'Вес Точности',
        'Enable Face Restorer 2': 'Включить Восстановление Лица 2',
        'Enable Face Expression Restorer': 'Включить Восстановление Выражения Лица',
        'Crop Scale': 'Масштаб Обрезки',
        'VY Ratio': 'Соотношение VY',
        'Expression Friendly Factor': 'Фактор Дружелюбности Выражения',
        'Animation Region': 'Область Анимации',
        'Normalize Lips': 'Нормализовать Губы',
        'Normalize Lips Threshold': 'Порог Нормализации Губ',
        'Retargeting Eyes': 'Перенацеливание Глаз',
        'Retargeting Eyes Multiplier': 'Множитель Перенацеливания Глаз',
        'Retargeting Lips': 'Перенацеливание Губ',
        'Retargeting Lips Multiplier': 'Множитель Перенацеливания Губ',
        'Swapper Model': 'Модель Замены',
        'Swapper Resolution': 'Разрешение Замены',
        'DFM Model': 'Модель DFM',
        'AMP Morph Factor': 'Фактор Морфинга AMP',
        'RCT Color Transfer': 'Перенос Цвета RCT',
        'Face Adjustments': 'Коррекция Лица',
        'Keypoints X-Axis': 'Ключевые точки по оси X',
        'Keypoints Y-Axis': 'Ключевые точки по оси Y',
        'Keypoints Scale': 'Масштаб Ключевых Точек',
        'Face Scale Amount': 'Масштаб Лица',
        '5 - Keypoints Adjustments': 'Коррекция 5 Ключевых Точек',
        'Left Eye:   X': 'Левый Глаз:   X',
        'Left Eye:   Y': 'Левый Глаз:   Y',
        'Right Eye:   X': 'Правый Глаз:   X',
        'Right Eye:   Y': 'Правый Глаз:   Y',
        'Nose:   X': 'Нос:   X',
        'Nose:   Y': 'Нос:   Y',
        'Left Mouth:   X': 'Левый Угол Рта:   X',
        'Left Mouth:   Y': 'Левый Угол Рта:   Y',
        'Right Mouth:   X': 'Правый Угол Рта:   X',
        'Right Mouth:   Y': 'Правый Угол Рта:   Y',
        'Similarity Threshold': 'Порог Сходства',
        'Strength': 'Сила',
        'Amount': 'Количество',
        'Face Likeness': 'Сходство Лиц',
        'Differencing': 'Различие',
        'Bottom Border': 'Нижняя Граница',
        'Left Border': 'Левая Граница',
        'Right Border': 'Правая Граница',
        'Top Border': 'Верхняя Граница',
        'Border Blur': 'Размытие Границы',
        'Occlusion Mask': 'Маска Окклюзии',
        'Size': 'Размер',
        'DFL XSeg Mask': 'Маска DFL XSeg',
        'Occluder/DFL XSeg Blur': 'Размытие Окклюдера/DFL XSeg',
        'Text Masking': 'Маскирование Текста',
        'Text Masking Entry': 'Ввод Маскирования Текста',
        'Face Parser Mask': 'Маска Парсера Лица',
        'Background': 'Фон',
        'Face': 'Лицо',
        'Left Eyebrow': 'Левая Бровь',
        'Right Eyebrow': 'Правая Бровь',
        'Left Eye': 'Левый Глаз',
        'Right Eye': 'Правый Глаз',
        'EyeGlasses': 'Очки',
        'Nose': 'Нос',
        'Mouth': 'Рот',
        'Upper Lip': 'Верхняя Губа',
        'Lower Lip': 'Нижняя Губа',
        'Neck': 'Шея',
        'Hair': 'Волосы',
        'Background Blur': 'Размытие Фона',
        'Face Blur': 'Размытие Лица',
        'Restore Eyes': 'Восстановить Глаза',
        'Eyes Blend Amount': 'Смешивание Глаз',
        'Eyes Size Factor': 'Фактор Размера Глаз',
        'Eyes Feather Blend': 'Плавное Смешивание Глаз',
        'X Eyes Radius Factor': 'Фактор Радиуса X Глаз',
        'Y Eyes Radius Factor': 'Фактор Радиуса Y Глаз',
        'X Eyes Offset': 'Смещение X Глаз',
        'Y Eyes Offset': 'Смещение Y Глаз',
        'Eyes Spacing Offset': 'Смещение Расстояния Глаз',
        'Restore Mouth': 'Восстановить Рот',
        'Mouth Blend Amount': 'Смешивание Рта',
        'Mouth Size Factor': 'Фактор Размера Рта',
        'Mouth Feather Blend': 'Плавное Смешивание Рта',
        'X Mouth Radius Factor': 'Фактор Радиуса X Рта',
        'Y Mouth Radius Factor': 'Фактор Радиуса Y Рта',
        'X Mouth Offset': 'Смещение X Рта',
        'Y Mouth Offset': 'Смещение Y Рта',
        'Eyes/Mouth Blur': 'Размытие Глаз/Рта',
        'AutoColor Transfer': 'Автоматический Перенос Цвета',
        'Transfer Type': 'Тип Переноса',
        'JPEG Compression': 'Сжатие JPEG',
        'Blend Adjustments': 'Настройки Смешивания',
        'Other Options': 'Другие Опции',
        'Face Editor': 'Редактор Лица',
        'Common': 'Общие',
        'Face Restorer': 'Восстановитель Лица',
        'Detector Settings': 'Настройки Детектора',
        'File': 'Файл',
        'Edit': 'Редактировать',
        'View': 'Вид',
        'Target Videos and Inputs': 'Целевые Видео и Входные Данные',
        'Target Videos/Images': 'Целевые Видео/Изображения',
        'Select Videos/Images Path': 'Выбрать Путь к Видео/Изображениям',
        'Search Videos/Images': 'Поиск Видео/Изображений',
        'Drop Files or Click here to Select a Folder': 'Перетащите файлы или нажмите для выбора папки',
        'Input Faces': 'Входные Лица',
        'Select Face Images Path': 'Выбрать Путь к Изображениям Лиц',
        'Search Faces': 'Поиск Лиц',
        'Save Image': 'Сохранить Изображение',
        'Find Faces': 'Найти Лица',
        'Clear Faces': 'Очистить Лица',
        'Swap Faces': 'Заменить Лица',
        'Edit Faces': 'Редактировать Лица',
        'Search Embeddings': 'Поиск Эмбеддингов',
        'Media Panel': 'Панель Медиа',
        'Faces Panel': 'Панель Лиц',
        'Parameters Panel': 'Панель Параметров',
        'View Face Compare': 'Сравнение Лиц',
        'View Face Mask': 'Маска Лица',
        'Control Options': 'Опции Управления',
        'Clear VRAM': 'Очистить VRAM',
        'Face Swap': 'Замена Лица',
        'Swapper': 'Заменщик',
        'Keypoints Adjustments': 'Коррекция Ключевых Точек',
        'Left Eye: X': 'Левый Глаз: X',
        'Left Eye: Y': 'Левый Глаз: Y',
        'Right Eye: X': 'Правый Глаз: X',
        'Right Eye: Y': 'Правый Глаз: Y',
        'Nose: X': 'Нос: X',
        'Nose: Y': 'Нос: Y',
        'Left Mouth: X': 'Левый Угол Рта: X',
        'Left Mouth: Y': 'Левый Угол Рта: Y',
        'Right Mouth: X': 'Правый Угол Рта: X',
        'Right Mouth: Y': 'Правый Угол Рта: Y',
        'Face Mask': 'Маска Лица',
        'Face Color Correction': 'Коррекция Цвета Лица',
        # Strings do LoadingProgressDialog
        'Carregando modelo': 'Загрузка модели',
        'Preparando para carregar modelo...': 'Подготовка к загрузке модели....',
        'Tempo decorrido': 'Прошедшее время',
        'Tempo estimado restante': 'Оставшееся расчетное время',
        'Carregamento concluído com sucesso!': 'Загрузка успешно завершена!',
        'Falha no carregamento do modelo!': 'Ошибка загрузки модели!',
        'Carregando modelo: {0}': 'Загрузка модели: {0}',
        'Cancelar': 'Отмена',
    },
    '中文': {
        'Configurações': '设置',
        'Preferências...': '偏好设置...',
        'Geral': '常规',
        'Desempenho': '性能',
        'Cache': '缓存',
        'Logs': '日志',
        'Idioma:': '语言:',
        'Tema:': '主题:',
        'Dispositivo de processamento:': '处理设备:',
        'Número de threads:': '线程数:',
        'Limite de VRAM (MB):': 'VRAM 限制 (MB):',
        'Formato do cache:': '缓存格式:',
        'Qualidade (JPEG/WebP):': '质量 (JPEG/WebP):',
        'Limpar cache de frames': '清除帧缓存',
        'Nível de log:': '日志级别:',
        'Salvar logs em arquivo': '保存日志到文件',
        'OK': '确定',
        'Cancelar': '取消',
        'Retroceder Frame': '后退帧',
        'Avançar Frame': '前进帧',
        'Reproduzir': '播放',
        'Gravar': '录制',
        'Adicionar Marcador': '添加标记',
        'Remover Marcador': '移除标记',
        'Próximo Marcador': '下一个标记',
        'Marcador Anterior': '上一个标记',
        'Ir para Frame': '跳转到帧',
        'Imagens': '图片',
        'Vídeos': '视频',
        'Webcams': '摄像头',
        'Uso de VRAM: %p%': '显存使用率: %p%',
        'Modelo de Segmentação': '分割模型',
        'Modelo de Detecção': '检测模型',
        'Dispositivo': '设备',
        'Limite de VRAM': '显存限制',
        'Formato do Cache': '缓存格式',
        'Qualidade do Cache': '缓存质量',
        'Face Landmarks Correction': '面部特征点校正',
        'Face Similarity:': '面部相似性',
        'Blur Amount:': '模糊程度',
        'Threads': '线程',
        'Salvar configurações': '保存设置',
        'Carregar configurações': '加载设置',
        'Resetar configurações': '重置设置',
        'Ajuda': '帮助',
        'Escolha o modelo de segmentação facial': '选择人脸分割模型',
        'Escolha o modelo de detecção facial': '选择人脸检测模型',
        'Selecione o dispositivo de processamento (CPU/GPU)': '选择处理设备 (CPU/GPU)',
        'Defina o limite de uso de VRAM para a GPU': '设置GPU显存使用限制',
        'Formato de compressão do cache de frames': '帧缓存压缩格式',
        'Qualidade da compressão do cache de frames': '帧缓存压缩质量',
        'Número de threads para processamento em lote': '批量处理线程数',
        'Salvar as configurações atuais em um arquivo': '将当前设置保存到文件',
        'Carregar configurações de um arquivo': '从文件加载设置',
        'Restaurar configurações padrão': '恢复默认设置',
        'Abrir documentação de ajuda': '打开帮助文档',
        'Red': '红色',
        'Green': '绿色',
        'Blue': '蓝色',
        'Blend Amount': '混合量',
        'Color Adjustments': '颜色调整',
        'Brightness': '亮度',
        'Contrast': '对比度',
        'Saturation': '饱和度',
        'Sharpness': '锐度',
        'Hue': '色调',
        'Gamma': '伽玛',
        'Noise': '噪声',
        'JPEG Compression': 'JPEG压缩',
        'Compression': '压缩',
        'Final Blend': '最终混合',
        'Final Blend Amount': '最终混合量',
        'Overall Mask Blend Amount': '整体蒙版混合量',
        'General': '常规',
        'Providers Priority': '提供者优先级',
        'Benchmark Threads': '基准线程',
        'Método de Segmentação de Rosto': '人脸分割方法',
        'Modelo SAM 2': 'SAM模型2',
        'Video Settings': '视频设置',
        'Set Custom Video Playback FPS': '设置视频播放FPS',
        'Video Playback FPS': '视频播放FPS',
        'Auto Swap': '自动交换',
        'Detectors': '检测器',
        'Face Detect Model': '人脸检测模型',
        'Detect Score': '检测分数',
        'Max No of Faces to Detect': '最大检测人脸数',
        'Auto Rotation': '自动旋转',
        'Manual Rotation': '手动旋转',
        'Rotation Angle': '旋转角度',
        'Enable Landmark Detection': '启用关键点检测',
        'Landmark Detect Model': '关键点检测模型',
        'Landmark Detect Score': '关键点检测分数',
        'Detect From Points': '从点检测',
        'Show Landmarks': '显示关键点',
        'Show Bounding Boxes': '显示边界框',
        'DFM Settings': 'DFM设置',
        'Maximum DFM Models to use': '最大DFM模型数',
        'Frame Enhancer': '帧增强器',
        'Enable Frame Enhancer': '启用帧增强器',
        'Frame Enhancer Type': '帧增强类型',
        'Blend': '混合',
        'Webcam Settings': '摄像头设置',
        'Webcam Max No': '最大摄像头数',
        'Webcam Backend': '摄像头后端',
        'Webcam Resolution': '摄像头分辨率',
        'Webcam FPS': '摄像头FPS',
        'Virtual Camera': '虚拟摄像头',
        'Send Frames to Virtual Camera': '发送帧到虚拟摄像头',
        'Virtual Camera Backend': '虚拟摄像头后端',
        'Face Recognition': '人脸识别',
        'Recognition Model': '识别模型',
        'Swapping Similarity Type': '交换相似度类型',
        'Embedding Merge Method': '嵌入合并方法',
        'Media Selection': '媒体选择',
        'Target Media Include Subfolders': '目标媒体包含子文件夹',
        'Input Faces Include Subfolders': '输入人脸包含子文件夹',
        'Enable Face Pose/Expression Editor': '启用人脸姿态/表情编辑器',
        'Face Editor Type': '人脸编辑器类型',
        'Eyes Close <--> Open Ratio': '眼睛闭合<-->睁开比例',
        'Lips Close <--> Open Ratio': '嘴唇闭合<-->张开比例',
        'Head Pitch': '头部俯仰',
        'Head Yaw': '头部偏航',
        'Head Roll': '头部滚转',
        'X-Axis Movement': 'X轴移动',
        'Y-Axis Movement': 'Y轴移动',
        'Z-Axis Movement': 'Z轴移动',
        'Mouth Pouting': '撅嘴',
        'Mouth Pursing': '抿嘴',
        'Mouth Grin': '咧嘴笑',
        'Lips Close <--> Open Value': '嘴唇闭合<-->张开值',
        'Mouth Smile': '微笑',
        'Eye Wink': '眨眼',
        'EyeBrows Direction': '眉毛方向',
        'EyeGaze Horizontal': '水平视线',
        'EyeGaze Vertical': '垂直视线',
        'Face Makeup': '面部化妆',
        'Hair Makeup': '头发化妆',
        'Lips Makeup': '嘴唇化妆',
        'EyeBrows Makeup': '眉毛化妆',
        'Enable Face Restorer': '启用人脸修复器',
        'Restorer Type': '修复器类型',
        'Alignment': '对齐',
        'Fidelity Weight': '保真度权重',
        'Enable Face Restorer 2': '启用人脸修复器 2',
        'Enable Face Expression Restorer': '启用人脸表情修复器',
        'Crop Scale': '裁剪比例',
        'VY Ratio': 'VY比例',
        'Expression Friendly Factor': '表情友好因子',
        'Animation Region': '动画区域',
        'Normalize Lips': '归一化嘴唇',
        'Normalize Lips Threshold': '归一化嘴唇阈值',
        'Retargeting Eyes': '眼睛重定向',
        'Retargeting Eyes Multiplier': '眼睛重定向乘数',
        'Retargeting Lips': '嘴唇重定向',
        'Retargeting Lips Multiplier': '嘴唇重定向乘数',
        'Swapper Model': '交换器模型',
        'Swapper Resolution': '交换器分辨率',
        'DFM Model': 'DFM模型',
        'AMP Morph Factor': 'AMP变形因子',
        'RCT Color Transfer': 'RCT颜色转移',
        'Face Adjustments': '人脸调整',
        'Keypoints X-Axis': '关键点X轴',
        'Keypoints Y-Axis': '关键点Y轴',
        'Keypoints Scale': '关键点比例',
        'Face Scale Amount': '人脸比例量',
        '5 - Keypoints Adjustments': '5 - 关键点调整',
        'Left Eye:   X': '左眼:   X',
        'Left Eye:   Y': '左眼:   Y',
        'Right Eye:   X': '右眼:   X',
        'Right Eye:   Y': '右眼:   Y',
        'Nose:   X': '鼻子:   X',
        'Nose:   Y': '鼻子:   Y',
        'Left Mouth:   X': '左嘴角:   X',
        'Left Mouth:   Y': '左嘴角:   Y',
        'Right Mouth:   X': '右嘴角:   X',
        'Right Mouth:   Y': '右嘴角:   Y',
        'Similarity Threshold': '相似度阈值',
        'Strength': '强度',
        'Amount': '量',
        'Face Likeness': '人脸相似度',
        'Differencing': '差异化',
        'Bottom Border': '底边框',
        'Left Border': '左边框',
        'Right Border': '右边框',
        'Top Border': '顶边框',
        'Border Blur': '边框模糊',
        'Occlusion Mask': '遮挡蒙版',
        'Size': '大小',
        'DFL XSeg Mask': 'DFL XSeg 蒙版',
        'Occluder/DFL XSeg Blur': '遮挡器/DFL XSeg 模糊',
        'Text Masking': '文本遮罩',
        'Text Masking Entry': '文本遮罩输入',
        'Face Parser Mask': '人脸解析蒙版',
        'Background': '背景',
        'Face': '人脸',
        'Left Eyebrow': '左眉毛',
        'Right Eyebrow': '右眉毛',
        'Left Eye': '左眼',
        'Right Eye': '右眼',
        'EyeGlasses': '眼镜',
        'Nose': '鼻子',
        'Mouth': '嘴巴',
        'Upper Lip': '上嘴唇',
        'Lower Lip': '下嘴唇',
        'Neck': '脖子',
        'Hair': '头发',
        'Background Blur': '背景模糊',
        'Face Blur': '人脸模糊',
        'Restore Eyes': '恢复眼睛',
        'Eyes Blend Amount': '眼睛混合量',
        'Eyes Size Factor': '眼睛大小因子',
        'Eyes Feather Blend': '眼睛羽化混合',
        'X Eyes Radius Factor': 'X轴眼睛半径因子',
        'Y Eyes Radius Factor': 'Y轴眼睛半径因子',
        'X Eyes Offset': 'X轴眼睛偏移',
        'Y Eyes Offset': 'Y轴眼睛偏移',
        'Eyes Spacing Offset': '眼睛间距偏移',
        'Restore Mouth': '恢复嘴巴',
        'Mouth Blend Amount': '嘴巴混合量',
        'Mouth Size Factor': '嘴巴大小因子',
        'Mouth Feather Blend': '嘴巴羽化混合',
        'X Mouth Radius Factor': 'X轴嘴巴半径因子',
        'Y Mouth Radius Factor': 'Y轴嘴巴半径因子',
        'X Mouth Offset': 'X轴嘴巴偏移',
        'Y Mouth Offset': 'Y轴嘴巴偏移',
        'Eyes/Mouth Blur': '眼睛/嘴巴模糊',
        'AutoColor Transfer': '自动颜色转移',
        'Transfer Type': '转移类型',
        'JPEG Compression': 'JPEG压缩',
        'Blend Adjustments': '混合调整',
        'Other Options': '其他选项',
        'Face Editor': '人脸编辑器',
        'Common': '通用',
        'Face Restorer': '人脸修复器',
        'Detector Settings': '检测器设置',
        'File': '文件',
        'Edit': '编辑',
        'View': '视图',
        'Target Videos and Inputs': '目标视频和输入',
        'Target Videos/Images': '目标视频/图片',
        'Select Videos/Images Path': '选择视频/图片路径',
        'Search Videos/Images': '搜索视频/图片',
        'Drop Files or Click here to Select a Folder': '拖放文件或点击此处选择文件夹',
        'Input Faces': '输入人脸',
        'Select Face Images Path': '选择人脸图片路径',
        'Search Faces': '搜索人脸',
        'Save Image': '保存图片',
        'Find Faces': '查找人脸',
        'Clear Faces': '清除人脸',
        'Swap Faces': '交换人脸',
        'Edit Faces': '编辑人脸',
        'Search Embeddings': '搜索嵌入',
        'Media Panel': '媒体面板',
        'Faces Panel': '人脸面板',
        'Parameters Panel': '参数面板',
        'View Face Compare': '查看人脸比较',
        'View Face Mask': '查看人脸蒙版',
        'Control Options': '控制选项',
        'Clear VRAM': '清除显存',
        'Face Swap': '人脸交换',
        'Swapper': '交换器',
        'Keypoints Adjustments': '关键点调整',
        'Left Eye: X': '左眼: X',
        'Left Eye: Y': '左眼: Y',
        'Right Eye: X': '右眼: X',
        'Right Eye: Y': '右眼: Y',
        'Nose: X': '鼻子: X',
        'Nose: Y': '鼻子: Y',
        'Left Mouth: X': '左嘴角: X',
        'Left Mouth: Y': '左嘴角: Y',
        'Right Mouth: X': '右嘴角: X',
        'Right Mouth: Y': '右嘴角: Y',
        'Face Mask': '人脸蒙版',
        'Face Color Correction': '人脸颜色校正',
        # LoadingProgressDialog 字符串
        'Carregando modelo': '加载模型',
        'Preparando para carregar modelo...': '准备加载模型...',
        'Tempo decorrido': '已用时间',
        'Tempo estimado restante': '预计剩余时间',
        'Carregamento concluído com sucesso!': '模型加载成功！',
        'Falha no carregamento do modelo!': '模型加载失败！',
        'Carregando modelo: {0}': '正在加载模型：{0}',
        'Cancelar': '取消',
    },
} # FECHAMENTO FINAL DO DICIONÁRIO PRINCIPAL
# Função para aplicar tema
def apply_theme(app, theme_text): # Renomeado para evitar conflito com self.tema
    app.setStyleSheet("")
    # Normalizar para inglês para comparação
    theme_map = {'Claro': 'Light', 'Escuro': 'Dark', 'Светлая': 'Light', 'Тёмная': 'Dark', '浅色': 'Light', '深色': 'Dark'}
    theme_normalized = theme_map.get(theme_text, 'Light') # Usa 'Light' se não encontrar

    if theme_normalized == 'Dark':
        app.setStyleSheet('''
            QWidget { background-color: #232629; color: #f0f0f0; font-family: Arial; font-size: 9pt; }
            QPushButton { background-color: #353535; color: #f0f0f0; border: 1px solid #555; padding: 5px; }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #252525; }
            QLineEdit, QComboBox, QSpinBox { background-color: #2c2c2c; color: #f0f0f0; border: 1px solid #555; padding: 3px; }
            QSlider::groove:horizontal { border: 1px solid #555; height: 8px; background: #2c2c2c; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #555; border: 1px solid #888; width: 18px; margin: -5px 0; border-radius: 9px; }
            QMenuBar { background-color: #232629; color: #f0f0f0; }
            QMenu { background-color: #2c2c2c; color: #f0f0f0; border: 1px solid #555; }
            QMenu::item:selected { background-color: #353535; }
            QTabWidget::pane { border-top: 1px solid #555; }
            QTabBar::tab { background: #2c2c2c; color: #f0f0f0; border: 1px solid #555; border-bottom: none; padding: 5px; }
            QTabBar::tab:selected { background: #353535; margin-bottom: -1px; }
            QTabBar::tab:!selected { margin-top: 2px; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; background-color: #232629; color: #f0f0f0;}
            QListWidget { background-color: #2c2c2c; color: #f0f0f0; border: 1px solid #555; }
            QDockWidget { background-color: #232629; color: #f0f0f0; }
            QDockWidget::title { background-color: #353535; text-align: left; padding: 5px; border: 1px solid #555; }
            QCheckBox { color: #f0f0f0; }
            QLabel { color: #f0f0f0; }
            QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; background-color: #2c2c2c; color: #f0f0f0; }
            QProgressBar::chunk { background-color: #05B8CC; width: 20px; }
            QToolTip { background-color: #353535; color: #f0f0f0; border: 1px solid #555; }
        ''')
    else: # Tema Claro ou padrão
        app.setStyleSheet('')

# Função para traduzir textos do SettingsDialog
def translate_settings_dialog(dlg, tr_func):
    dlg.setWindowTitle(tr_func('Configurações')) 
    dlg.tabs.setTabText(0, tr_func('Geral'))
    dlg.tabs.setTabText(1, tr_func('Desempenho'))
    dlg.tabs.setTabText(2, tr_func('Cache'))
    dlg.tabs.setTabText(3, tr_func('Logs'))
    
    # Traduzir textos dentro das abas
    try: # Geral
        dlg.labelIdioma.setText(tr_func('Idioma:')) 
        dlg.labelTema.setText(tr_func('Tema:'))
        dlg.labelOutputDir.setText(tr_func('Diretório de Saída:'))
        dlg.output_dir_btn.setText(tr_func('Selecionar Pasta'))
    except AttributeError as e: print(f"Warn: Widget não encontrado na aba Geral (SettingsDialog): {e}")
    except Exception as e: print(f"Erro ao traduzir aba Geral (SettingsDialog): {e}")

    try: # Desempenho
        dlg.labelDevice.setText(tr_func('Dispositivo de processamento:'))
        dlg.labelThreads.setText(tr_func('Número de threads:'))
        dlg.labelVramLimit.setText(tr_func('Limite de VRAM (MB):'))
    except AttributeError as e: print(f"Warn: Widget não encontrado na aba Desempenho (SettingsDialog): {e}")
    except Exception as e: print(f"Erro ao traduzir aba Desempenho (SettingsDialog): {e}")

    try: # Cache
        dlg.labelCacheFormat.setText(tr_func('Formato do cache:'))
        dlg.labelCacheQuality.setText(tr_func('Qualidade (JPEG/WebP):'))
        dlg.clear_cache_btn.setText(tr_func('Limpar cache de frames'))
    except AttributeError as e: print(f"Warn: Widget não encontrado na aba Cache (SettingsDialog): {e}")
    except Exception as e: print(f"Erro ao traduzir aba Cache (SettingsDialog): {e}")

    try: # Logs
        dlg.labelLogLevel.setText(tr_func('Nível de log:'))
        dlg.save_logs_checkbox.setText(tr_func('Salvar logs em arquivo'))
    except AttributeError as e: print(f"Warn: Widget não encontrado na aba Logs (SettingsDialog): {e}")
    except Exception as e: print(f"Erro ao traduzir aba Logs (SettingsDialog): {e}")
        
    # Botões
    try:
        if hasattr(dlg, 'okButton'): dlg.okButton.setText(tr_func('OK'))
        if hasattr(dlg, 'cancelButton'): dlg.cancelButton.setText(tr_func('Cancelar'))
    except Exception as e: print(f"Erro ao traduzir botões (SettingsDialog): {e}")


class CacheSettingsDialog(QDialog):
    def __init__(self, parent=None, main_window=None, current_format='webp', current_quality=90): # Adicionado main_window
        super().__init__(parent)
        self.main_window = main_window
        self.tr_func = getattr(self.main_window, 'tr', lambda x: x) # Usar tr da main_window
        
        self.setWindowTitle(self.tr_func('Configurações de Cache'))
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(self.tr_func('Formato do cache:'))) 
        self.format_combo = QComboBox()
        self.format_combo.addItems(['webp', 'jpeg', 'png'])
        self.format_combo.setCurrentText(current_format)
        layout.addWidget(self.format_combo)

        layout.addWidget(QLabel(self.tr_func('Qualidade (JPEG/WebP):'))) 
        self.quality_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.quality_slider.setMinimum(50); self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(current_quality)
        layout.addWidget(self.quality_slider)
        self.quality_label = QLabel(str(current_quality))
        layout.addWidget(self.quality_label)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(str(v)))

        def update_quality_enabled(): self.quality_slider.setEnabled(self.format_combo.currentText() in ['jpeg', 'webp'])
        self.format_combo.currentTextChanged.connect(update_quality_enabled)
        update_quality_enabled()

        self.clear_cache_btn = QPushButton(self.tr_func('Limpar cache de frames')) 
        layout.addWidget(self.clear_cache_btn)
        self.clear_cache_btn.clicked.connect(self.clear_cache)

        btns = QHBoxLayout()
        ok_btn = QPushButton(self.tr_func('OK')); cancel_btn = QPushButton(self.tr_func('Cancelar'))
        btns.addWidget(ok_btn); btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)

    def clear_cache(self):
        from app.processors.video_processor import CACHE_BASE_DIR; import shutil
        cache_dir = Path(CACHE_BASE_DIR)
        if cache_dir.exists() and cache_dir.is_dir():
            try: shutil.rmtree(cache_dir); QMessageBox.information(self, self.tr_func('Cache limpo'), self.tr_func('Cache de frames removido com sucesso!'))
            except Exception as e: QMessageBox.warning(self, self.tr_func('Erro ao Limpar Cache'), f"{self.tr_func('Não foi possível remover o diretório de cache:')}\n{e}")
        else: QMessageBox.information(self, self.tr_func('Cache limpo'), self.tr_func('Nenhum cache encontrado.'))

    def get_settings(self):
        return {'format': self.format_combo.currentText(), 'quality': self.quality_slider.value()}


class SettingsDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.tr_func = getattr(main_window, 'tr', lambda x: x) 
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Criação das Abas (usando self.tr_func para textos iniciais)
        # Aba Geral
        tab_geral = QtWidgets.QWidget(); geral_layout = QVBoxLayout(tab_geral)
        self.labelIdioma = QLabel(); self.labelIdioma.setObjectName("labelIdioma"); geral_layout.addWidget(self.labelIdioma)
        self.idioma_combo = QComboBox(); self.idioma_combo.addItems(['Português', 'English', 'Русский', '中文']); geral_layout.addWidget(self.idioma_combo)
        self.labelTema = QLabel(); self.labelTema.setObjectName("labelTema"); geral_layout.addWidget(self.labelTema)
        self.tema_combo = QComboBox(); self.tema_combo.addItems(['Claro', 'Escuro']); geral_layout.addWidget(self.tema_combo)
        self.labelOutputDir = QLabel(); self.labelOutputDir.setObjectName("labelOutputDir"); geral_layout.addWidget(self.labelOutputDir)
        dir_layout = QHBoxLayout(); self.output_dir_line = QLineEdit(); self.output_dir_line.setReadOnly(True); dir_layout.addWidget(self.output_dir_line)
        self.output_dir_btn = QPushButton(); self.output_dir_btn.setObjectName("outputDirBtn"); dir_layout.addWidget(self.output_dir_btn)
        self.output_dir_btn.clicked.connect(self.select_output_dir); geral_layout.addLayout(dir_layout)
        self.tabs.addTab(tab_geral, self.tr_func('General'))

        # Aba Desempenho
        tab_perf = QtWidgets.QWidget(); perf_layout = QVBoxLayout(tab_perf)
        self.labelDevice = QLabel(); self.labelDevice.setObjectName("labelDevice"); perf_layout.addWidget(self.labelDevice)
        self.device_combo = QComboBox(); self.device_combo.addItems(['Auto', 'CPU', 'GPU 0', 'GPU 1']); perf_layout.addWidget(self.device_combo)
        self.labelThreads = QLabel(); self.labelThreads.setObjectName("labelThreads"); perf_layout.addWidget(self.labelThreads)
        self.threads_spin = QSpinBox(); self.threads_spin.setMinimum(1); self.threads_spin.setMaximum(os.cpu_count() or 32); self.threads_spin.setEnabled(True); perf_layout.addWidget(self.threads_spin)
        self.labelVramLimit = QLabel(); self.labelVramLimit.setObjectName("labelVramLimit"); perf_layout.addWidget(self.labelVramLimit)
        self.vram_slider = QSlider(QtCore.Qt.Orientation.Horizontal); self.vram_slider.setMinimum(512); self.vram_slider.setMaximum(32768); perf_layout.addWidget(self.vram_slider)
        self.vram_label = QLabel(); self.vram_label.setObjectName("vramLabel"); perf_layout.addWidget(self.vram_label)
        self.vram_slider.valueChanged.connect(lambda v: self.vram_label.setText(str(v) + ' MB'))
        self.tabs.addTab(tab_perf, self.tr_func('Desempenho'))

        # Aba Cache
        tab_cache = QtWidgets.QWidget(); cache_layout = QVBoxLayout(tab_cache)
        self.labelCacheFormat = QLabel(); self.labelCacheFormat.setObjectName("labelCacheFormat"); cache_layout.addWidget(self.labelCacheFormat)
        self.format_combo = QComboBox(); self.format_combo.addItems(['webp', 'jpeg', 'png']); cache_layout.addWidget(self.format_combo)
        self.labelCacheQuality = QLabel(); self.labelCacheQuality.setObjectName("labelCacheQuality"); cache_layout.addWidget(self.labelCacheQuality)
        self.quality_slider = QSlider(QtCore.Qt.Orientation.Horizontal); self.quality_slider.setMinimum(50); self.quality_slider.setMaximum(100); cache_layout.addWidget(self.quality_slider)
        self.quality_label = QLabel(); self.quality_label.setObjectName("qualityLabel"); cache_layout.addWidget(self.quality_label)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(str(v)))
        def update_quality_enabled(): self.quality_slider.setEnabled(self.format_combo.currentText() in ['jpeg', 'webp'])
        self.format_combo.currentTextChanged.connect(update_quality_enabled); update_quality_enabled()
        self.clear_cache_btn = QPushButton(); self.clear_cache_btn.setObjectName("clearCacheBtn"); cache_layout.addWidget(self.clear_cache_btn)
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        self.tabs.addTab(tab_cache, self.tr_func('Cache'))

        # Aba Logs
        tab_logs = QtWidgets.QWidget(); logs_layout = QVBoxLayout(tab_logs)
        self.labelLogLevel = QLabel(); self.labelLogLevel.setObjectName("labelLogLevel"); logs_layout.addWidget(self.labelLogLevel)
        self.loglevel_combo = QComboBox(); self.loglevel_combo.addItems(['Silencioso', 'Informativo', 'Debug']); logs_layout.addWidget(self.loglevel_combo)
        self.save_logs_checkbox = QCheckBox(); self.save_logs_checkbox.setObjectName("saveLogsCheckbox"); logs_layout.addWidget(self.save_logs_checkbox)
        self.tabs.addTab(tab_logs, self.tr_func('Logs'))

        # Botões OK/Cancelar
        btns_layout = QHBoxLayout() 
        self.okButton = QPushButton(); self.okButton.setObjectName("okButton"); 
        self.cancelButton = QPushButton(); self.cancelButton.setObjectName("cancelButton")
        btns_layout.addStretch(); btns_layout.addWidget(self.okButton); btns_layout.addWidget(self.cancelButton)
        layout.addLayout(btns_layout) 
        self.okButton.clicked.connect(self.accept); self.cancelButton.clicked.connect(self.reject)

    def set_settings(self, settings):
        """Define os valores dos widgets com base nas configurações atuais da MainWindow."""
        self.idioma_combo.setCurrentText(settings.get('idioma', 'Português'))
        self.tema_combo.setCurrentText(settings.get('tema', 'Claro'))
        self.output_dir_line.setText(settings.get('output_dir', ''))
        self.device_combo.setCurrentText(settings.get('device', 'Auto'))
        self.threads_spin.setValue(settings.get('num_threads', 2))
        self.vram_slider.setValue(settings.get('vram_limit', 4096))
        self.vram_label.setText(str(settings.get('vram_limit', 4096)) + ' MB') 
        self.format_combo.setCurrentText(settings.get('cache_format', 'webp'))
        self.quality_slider.setValue(settings.get('cache_quality', 90))
        self.quality_label.setText(str(settings.get('cache_quality', 90))) 
        self.quality_slider.setEnabled(self.format_combo.currentText() in ['jpeg', 'webp']) 
        log_level_map_inv = {'Silencioso': 0, 'Informativo': 1, 'Debug': 2}
        self.loglevel_combo.setCurrentIndex(log_level_map_inv.get(settings.get('loglevel', 'Informativo'), 1))
        self.save_logs_checkbox.setChecked(settings.get('save_logs', False))

    def clear_cache(self):
        # Reutiliza a lógica da MainWindow se possível, ou reimplementa
        if self.main_window and hasattr(self.main_window, 'clear_cache_action') and callable(self.main_window.clear_cache_action):
            self.main_window.clear_cache_action() # Exemplo, se existir uma ação na main window
        else:
            from app.processors.video_processor import CACHE_BASE_DIR; import shutil
            cache_dir = Path(CACHE_BASE_DIR)
            if cache_dir.exists() and cache_dir.is_dir():
                try: shutil.rmtree(cache_dir); QMessageBox.information(self, self.tr_func('Cache limpo'), self.tr_func('Cache de frames removido com sucesso!'))
                except Exception as e: QMessageBox.warning(self, self.tr_func('Erro ao Limpar Cache'), f"{self.tr_func('Não foi possível remover o diretório de cache:')}\n{e}")
            else: QMessageBox.information(self, self.tr_func('Cache limpo'), self.tr_func('Nenhum cache encontrado.'))

    def select_output_dir(self):
        start_dir = self.output_dir_line.text() if self.output_dir_line.text() else str(Path.home()) # Usar Path.home()
        dir_path = QFileDialog.getExistingDirectory(self, self.tr_func('Selecionar Diretório de Saída'), start_dir)
        if dir_path: self.output_dir_line.setText(dir_path)

    def get_settings(self):
        log_level_map = {0: 'Silencioso', 1: 'Informativo', 2: 'Debug'}
        return {
            'idioma': self.idioma_combo.currentText(),
            'tema': self.tema_combo.currentText(),
            'device': self.device_combo.currentText(),
            'num_threads': self.threads_spin.value(),
            'vram_limit': self.vram_slider.value(),
            'output_dir': self.output_dir_line.text(),
            'cache_format': self.format_combo.currentText(),
            'cache_quality': self.quality_slider.value(),
            'loglevel': log_level_map.get(self.loglevel_combo.currentIndex(), 'Informativo'),
            'save_logs': self.save_logs_checkbox.isChecked(),
        }


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def __init__(self):
        super(MainWindow, self).__init__()
        # Definir atributos padrão ANTES de carregar config
        self.idioma = 'Português' 
        self.tema = 'Claro' 
        self.dynamic_widgets = [] # Inicializar lista ANTES de qualquer coisa

        self.load_config() # Carregar config que pode sobrescrever os padrões
        self.setupUi(self)
        self.initialize_variables() # Inicializa outras variáveis e processadores
        self.initialize_widgets() # Configura widgets, conecta sinais, chama add_widgets_to_tab_layout
        self.load_last_workspace()
        
        # Aplicar tema e idioma ao iniciar (USA OS VALORES FINAIS de self.tema e self.idioma)
        print(f"[DEBUG][__init__] Aplicando TEMA inicial: {self.tema}")
        apply_theme(QtWidgets.QApplication.instance(), self.tema) 
        print(f"[DEBUG][__init__] Aplicando IDIOMA inicial: {self.idioma}")
        self.apply_language(self.idioma) 

    # FUNÇÃO tr DEFINIDA AQUI
    def tr(self, text):
        """Retorna a tradução do texto para o idioma atual ou o próprio texto."""
        idioma_usado_para_busca = getattr(self, 'idioma', 'Português') 
        dicionario_idioma = TRANSLATIONS.get(idioma_usado_para_busca, TRANSLATIONS.get('Português', {}))
        traduzido = dicionario_idioma.get(text, text) 
        print(f"[DEBUG][tr] Idioma='{idioma_usado_para_busca}', Chave='{text}' -> Traduzido='{traduzido}'")
        return traduzido

    # FUNÇÃO register_dynamic_widget DEFINIDA AQUI
    def register_dynamic_widget(self, widget, translation_key):
        """Registra um widget para ter seu texto/tooltip atualizado dinamicamente."""
        if not hasattr(self, 'dynamic_widgets'): self.dynamic_widgets = []
        widget.setProperty("translation_key", translation_key) 
        self.dynamic_widgets.append(widget)
        # Debug
        # widget_name_for_log = widget.objectName() if hasattr(widget, 'objectName') and widget.objectName() else str(type(widget))
        # print(f"[DEBUG][register_dynamic_widget] Registrado: {widget_name_for_log} com key='{translation_key}' | Total: {len(self.dynamic_widgets)}")

    # FUNÇÃO apply_language DEFINIDA AQUI
    def apply_language(self, idioma):
        print(f"\n{'='*10} Iniciando apply_language para IDIOMA='{idioma}' {'='*10}")
        registered_widget_count = len(self.dynamic_widgets) if hasattr(self, 'dynamic_widgets') else 0
        print(f"[DEBUG][apply_language INÍCIO] self.idioma foi definido como '{self.idioma}'. Widgets registrados = {registered_widget_count}")
        
        # Atualizar elementos estáticos (menus, abas principais)
        menubar = self.menuBar()
        try:
            if menubar and menubar.actions():
                actions = menubar.actions()
                if len(actions) > 0: actions[0].setText(self.tr('File')) # Usar chave INGLÊS como canônica
                if len(actions) > 1: actions[1].setText(self.tr('Edit'))  
                if len(actions) > 2: actions[2].setText(self.tr('View'))
            if hasattr(self, 'menu_configuracoes'): self.menu_configuracoes.setTitle(self.tr('Configurações'))
            if hasattr(self, 'action_preferencias'): self.action_preferencias.setText(self.tr('Preferências...'))
        except Exception as e: print(f"[ERROR][apply_language] Erro ao traduzir menus: {e}")

        try: # Abas Principais
            if hasattr(self, 'tabWidget') and self.tabWidget:
                 tab_map = {
                     self.face_swap_tab: 'Face Swap', self.face_editor_tab: 'Face Editor',
                     self.common_tab: 'Common', self.settings_tab: 'Detector Settings' }
                 for i in range(self.tabWidget.count()):
                     widget = self.tabWidget.widget(i)
                     if widget in tab_map:
                         chave_original = tab_map[widget]
                         self.tabWidget.setTabText(i, self.tr(chave_original))
        except Exception as e: print(f"[ERROR][apply_language] Erro ao traduzir abas principais: {e}")

        # Atualizar widgets dinâmicos registrados
        if hasattr(self, 'dynamic_widgets') and self.dynamic_widgets:
            print(f"[DEBUG][apply_language LOOP] Iniciando loop sobre {len(self.dynamic_widgets)} widgets dinâmicos...")
            for index, widget in enumerate(self.dynamic_widgets):
                widget_name_for_log = widget.objectName() if hasattr(widget,'objectName') and widget.objectName() else str(type(widget))
                try:
                    translation_key = widget.property("translation_key") 
                    # Print especial para widgets problemáticos
                    if translation_key in [
                        'Face Landmarks Correction', 'FaceSimilarity', 'Blur Amount',
                        'Target Videos/Images', 'Select Videos/Images Path', 'Search Videos/Images',
                        'Drop Files or Click here to Select a Folder', 'Input Faces', 'Select Face Images Path',
                        'Search Faces', 'Save Image', 'Find Faces', 'Clear Faces', 'Swap Faces', 'Edit Faces',
                        'Search Embeddings', 'Media Panel', 'Faces Panel', 'Parameters Panel', 'View Face Compare',
                        'View Face Mask', 'Clear VRAM', 'Control Options']:
                        print(f"[DEBUG][apply_language] VERIFICANDO WIDGET PROBLEMÁTICO: Key='{translation_key}' | Widget: {widget_name_for_log}")
                    if translation_key: 
                        translated_text = self.tr(translation_key) 
                        updated = False
                        if hasattr(widget, 'setText') and callable(widget.setText):
                            if not isinstance(widget, ParameterResetDefaultButton):
                                widget.setText(translated_text); updated = True
                        if hasattr(widget, 'setTitle') and callable(widget.setTitle):
                             widget.setTitle(translated_text); updated = True
                        if hasattr(widget, 'setToolTip') and callable(widget.setToolTip):
                             tooltip_traduzido = self.tr(translation_key) 
                             if tooltip_traduzido != translation_key: 
                                  widget.setToolTip(tooltip_traduzido)
                except Exception as e: print(f"  [LOOP {index+1}] Widget: {widget_name_for_log} -> ERRO AO ATUALIZAR: {e}")
        else: print("[DEBUG][apply_language] Nenhum widget dinâmico registrado.")

        # Atualização manual adicional de widgets estáticos/problemáticos
        try:
            # DockWidgets e GroupBoxes
            if hasattr(self, 'input_Target_DockWidget'): self.input_Target_DockWidget.setWindowTitle(self.tr('Target Videos and Inputs'))
            if hasattr(self, 'groupBox_TargetVideos_Select'): self.groupBox_TargetVideos_Select.setTitle(self.tr('Target Videos/Images'))
            if hasattr(self, 'groupBox_InputFaces_Select'): self.groupBox_InputFaces_Select.setTitle(self.tr('Input Faces'))
            if hasattr(self, 'controlOptionsDockWidget'): self.controlOptionsDockWidget.setWindowTitle(self.tr('Control Options'))
            if hasattr(self, 'groupBox_FaceLandmarksCorrection'): self.groupBox_FaceLandmarksCorrection.setTitle(self.tr('Face Landmarks Correction'))
            if hasattr(self, 'groupBox_FaceSimilarity'): self.groupBox_FaceSimilarity.setTitle(self.tr('Face Similarity'))
            # Labels
            if hasattr(self, 'labelTargetVideosPath'): self.labelTargetVideosPath.setText(self.tr('Select Videos/Images Path'))
            if hasattr(self, 'labelInputFacesPath'): self.labelInputFacesPath.setText(self.tr('Select Face Images Path'))
            if hasattr(self, 'targetVideosSearchBox'): self.targetVideosSearchBox.setPlaceholderText(self.tr('Search Videos/Images'))
            if hasattr(self, 'inputFacesSearchBox'): self.inputFacesSearchBox.setPlaceholderText(self.tr('Search Faces'))
            if hasattr(self, 'saveImageButton'): self.saveImageButton.setText(self.tr('Save Image'))
            if hasattr(self, 'findTargetFacesButton'): self.findTargetFacesButton.setText(self.tr('Find Faces'))
            if hasattr(self, 'clearTargetFacesButton'): self.clearTargetFacesButton.setText(self.tr('Clear Faces'))
            if hasattr(self, 'swapfacesButton'): self.swapfacesButton.setText(self.tr('Swap Faces'))
            if hasattr(self, 'editFacesButton'): self.editFacesButton.setText(self.tr('Edit Faces'))
            if hasattr(self, 'inputEmbeddingsSearchBox'): self.inputEmbeddingsSearchBox.setPlaceholderText(self.tr('Search Embeddings'))
            if hasattr(self, 'mediaPanelCheckBox'): self.mediaPanelCheckBox.setText(self.tr('Media Panel'))
            if hasattr(self, 'facesPanelCheckBox'): self.facesPanelCheckBox.setText(self.tr('Faces Panel'))
            if hasattr(self, 'parametersPanelCheckBox'): self.parametersPanelCheckBox.setText(self.tr('Parameters Panel'))
            if hasattr(self, 'faceCompareCheckBox'): self.faceCompareCheckBox.setText(self.tr('View Face Compare'))
            if hasattr(self, 'faceMaskCheckBox'): self.faceMaskCheckBox.setText(self.tr('View Face Mask'))
            if hasattr(self, 'clearMemoryButton'): self.clearMemoryButton.setText(self.tr('Clear VRAM'))
            # Labels estáticos específicos
            if hasattr(self, 'faceSimilarityLabel'): self.faceSimilarityLabel.setText(self.tr('Face Similarity'))
            if hasattr(self, 'faceLandmarksLabel'): self.faceLandmarksLabel.setText(self.tr('Face Landmarks Correction'))
            if hasattr(self, 'blurAmountLabel'): self.blurAmountLabel.setText(self.tr('Blur Amount'))
            # Placeholders de QListWidget (não suportado nativamente)
            # Se o placeholder for um QLabel filho, tente encontrar e traduzir:
            # Exemplo: placeholder_label = self.targetVideosList.findChild(QLabel, 'placeholderLabel')
            # if placeholder_label: placeholder_label.setText(self.tr('Drop Files or Click here to Select a Folder'))
            # Caso não seja possível, adicionar comentário:
            # TODO: Traduzir placeholder de QListWidget se implementado via QLabel filho.
        except Exception as e: print(f"[ERROR][apply_language] Erro ao traduzir estáticos manuais: {e}")

        print(f"{'='*10} Finalizado apply_language para IDIOMA='{self.idioma}' {'='*10}\n")
        
    def initialize_variables(self):
        self.video_loader_worker: Optional[ui_workers.TargetMediaLoaderWorker] = None
        self.input_faces_loader_worker: Optional[ui_workers.InputFacesLoaderWorker] = None
        self.target_videos_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='target_videos')
        self.input_faces_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='input_faces')
        self.merged_embeddings_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
        self.video_processor = VideoProcessor(self)
        self.loading_progress_dialog = LoadingProgressDialog(self)
        self.models_processor = ModelsProcessor(self)
        self.models_processor.start_loading_signal.connect(self.loading_progress_dialog.start_loading)
        self.models_processor.progress_signal.connect(self.loading_progress_dialog.update_progress)
        self.models_processor.completion_signal.connect(self.loading_progress_dialog.finish_loading)
        self.target_videos: Dict[int, widget_components.TargetMediaCardButton] = {}
        self.target_faces: Dict[int, widget_components.TargetFaceCardButton] = {}
        self.input_faces: Dict[int, widget_components.InputFaceCardButton] = {}
        self.merged_embeddings: Dict[int, widget_components.EmbeddingCardButton] = {}
        self.cur_selected_target_face_button: Optional[widget_components.TargetFaceCardButton] = None 
        self.selected_video_button: Optional[widget_components.TargetMediaCardButton] = None 
        self.selected_target_face_id: Optional[int] = None 
        self.parameters: FacesParametersTypes = {} 
        self.default_parameters: ParametersTypes = {}
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}
        self.markers: MarkerTypes = {} 
        self.parameters_list = {}
        self.control: ControlTypes = {}
        self.parameter_widgets: ParametersWidgetTypes = {}
        self.loaded_embedding_filename: str = ''
        self.last_target_media_folder_path = ''
        self.last_input_media_folder_path = ''
        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA
        self.loading_new_media = False
        self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
        self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))
        self.model_loading_signal.connect(partial(common_widget_actions.show_model_loading_dialog, self))
        self.model_loaded_signal.connect(partial(common_widget_actions.hide_model_loading_dialog, self))
        self.display_messagebox_signal.connect(partial(common_widget_actions.create_and_show_messagebox, self))

    def initialize_widgets(self):
        self.targetVideosList.setFlow(QtWidgets.QListWidget.Flow.LeftToRight)
        self.targetVideosList.setWrapping(True); self.targetVideosList.setResizeMode(QtWidgets.QListWidget.ResizeMode.Adjust)
        self.inputFacesList.setFlow(QtWidgets.QListWidget.Flow.LeftToRight)
        self.inputFacesList.setWrapping(True); self.inputFacesList.setResizeMode(QtWidgets.QListWidget.ResizeMode.Adjust)
        layout_actions.set_up_menu_actions(self)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)
        self.targetVideosList.setAcceptDrops(True); self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True); self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter); self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter); self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)
        self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
        self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))
        self.scene = QtWidgets.QGraphicsScene(self); self.graphicsViewFrame.setScene(self.scene)
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)
        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)
        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
        self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))
        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))
        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
        self.findTargetFacesButton.clicked.connect(partial(card_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filter_target_videos, self))
        self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterWebcamsCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterWebcamsCheckBox.clicked.connect(partial(list_view_actions.load_target_webcams, self))
        self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filter_input_faces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filter_merged_embeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(save_load_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self, True))
        self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))
        self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))
        self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))
        self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
        self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
        self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))
        self.faceMaskCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        self.faceCompareCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        
        # Criar widgets dinâmicos (CHAMA register_dynamic_widget internamente)
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=COMMON_LAYOUT_DATA, layoutWidget=self.commonWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

        # Tentar encontrar o botão pelo nome do objeto (ajuste o nome se diferente no seu UI)
        self.outputFolderButton = self.settings_tab.findChild(QPushButton, "outputFolderButton") 
        if self.outputFolderButton:
             self.outputFolderButton.clicked.connect(partial(list_view_actions.select_output_media_folder, self))
             # Registrar para tradução se necessário (assumindo que o texto é 'Selecionar Pasta')
             self.register_dynamic_widget(self.outputFolderButton, 'Selecionar Pasta')
        else: print("[WARN] Botão 'outputFolderButton' não encontrado.") 

        common_widget_actions.create_control(self, 'OutputMediaFolder', '') 
        self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)
        video_control_actions.reset_media_buttons(self)
        font = self.vramProgressBar.font(); font.setBold(True); self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        self.tabWidget.setCurrentIndex(0)
        
        # Configurar menus depois que self.tr está disponível
        self.menu_configuracoes = self.menuBar().addMenu(self.tr('Configurações')) 
        self.action_preferencias = QtGui.QAction(self.tr('Preferências...'), self) 
        self.menu_configuracoes.addAction(self.action_preferencias)
        self.action_preferencias.triggered.connect(self.open_settings_dialog)
        
        registered_widget_count_final = len(self.dynamic_widgets) if hasattr(self, 'dynamic_widgets') else 0
        print(f"[DEBUG][initialize_widgets FIM] Total final registrados: {registered_widget_count_final}")


    # --- Métodos restantes da MainWindow ---
    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if hasattr(self, 'scene') and self.scene and self.scene.items():
             pixmap_item = self.scene.items()[0]
             if isinstance(pixmap_item, QtWidgets.QGraphicsPixmapItem): # Verificar tipo
                 scene_rect = pixmap_item.boundingRect()
                 self.graphicsViewFrame.setSceneRect(scene_rect)
                 graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect )

    def keyPressEvent(self, event: QtGui.QKeyEvent): # Adicionado tipo
        match event.key():
            case QtCore.Qt.Key.Key_F11: video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key.Key_V: video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key.Key_C: video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key.Key_D: video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key.Key_A: video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key.Key_Z: self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key.Key_Space: self.buttonMediaPlay.click()
            case QtCore.Qt.Key.Key_R: self.buttonMediaRecord.click()
            case QtCore.Qt.Key.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier: video_control_actions.remove_video_slider_marker(self)
                else: video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key.Key_W: video_control_actions.move_slider_to_next_nearest_marker(self) # 'next' é o padrão
            case QtCore.Qt.Key.Key_Q: video_control_actions.move_slider_to_previous_nearest_marker(self)
            case QtCore.Qt.Key.Key_S: self.swapfacesButton.click()
            case _: super().keyPressEvent(event) # Passar eventos não tratados

    def closeEvent(self, event: QtGui.QCloseEvent): # Adicionado tipo
        print("MainWindow: closeEvent called.")
        if hasattr(self, 'video_processor'): self.video_processor.stop_processing()
        if hasattr(self, 'input_faces_loader_worker') and self.input_faces_loader_worker: list_view_actions.clear_stop_loading_input_media(self)
        if hasattr(self, 'video_loader_worker') and self.video_loader_worker: list_view_actions.clear_stop_loading_target_media(self)
        save_load_actions.save_current_workspace(self, 'last_workspace.json')
        self.save_config() 
        event.accept()

    def load_last_workspace(self):
        if Path('last_workspace.json').is_file():
            try:
                load_dialog = widget_components.LoadLastWorkspaceDialog(self)
                load_dialog.exec() 
            except Exception as e:
                print(f"Erro ao tentar carregar último workspace: {e}")

    def load_config(self):
        config = {}
        config_path = Path(CONFIG_FILE)
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Erro ao carregar config.json: {e}. Usando padrões.")
        
        # Define os atributos com valor padrão PRIMEIRO
        self.cache_format = 'webp'
        self.cache_quality = 90
        self.num_threads = os.cpu_count() // 2 or 2
        self.vram_limit = 4096
        self.device = 'Auto'
        self.idioma = 'Português'
        self.tema = 'Claro'
        self.loglevel = 'Informativo'
        self.save_logs = False
        self.output_dir = str(Path.home())

        # Atualiza com valores do config, se existirem
        self.cache_format = config.get('cache_format', self.cache_format)
        self.cache_quality = config.get('cache_quality', self.cache_quality)
        self.num_threads = config.get('num_threads', self.num_threads)
        self.vram_limit = config.get('vram_limit', self.vram_limit)
        self.device = config.get('device', self.device)
        self.idioma = config.get('idioma', self.idioma) 
        self.tema = config.get('tema', self.tema) 
        self.loglevel = config.get('loglevel', self.loglevel)
        self.save_logs = config.get('save_logs', self.save_logs)
        self.output_dir = config.get('output_dir', self.output_dir)
        if not self.output_dir: # Garante que não fique vazio
             self.output_dir = str(Path.home())


    def save_config(self):
        config = {
            'cache_format': getattr(self, 'cache_format', 'webp'),
            'cache_quality': getattr(self, 'cache_quality', 90),
            'num_threads': getattr(self, 'num_threads', 2),
            'vram_limit': getattr(self, 'vram_limit', 4096),
            'device': getattr(self, 'device', 'Auto'),
            'idioma': getattr(self, 'idioma', 'Português'),
            'tema': getattr(self, 'tema', 'Claro'),
            'loglevel': getattr(self, 'loglevel', 'Informativo'),
            'save_logs': getattr(self, 'save_logs', False),
            'output_dir': getattr(self, 'output_dir', str(Path.home())),
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erro ao salvar config.json: {e}")
            
    def open_settings_dialog(self):
        print("[DEBUG] Abrindo SettingsDialog...")
        dlg = SettingsDialog(parent=self, main_window=self) 
        current_settings = self.get_current_settings_for_dialog()
        print(f"[DEBUG] Configurações atuais para o diálogo: {current_settings}")
        dlg.set_settings(current_settings) 
        translate_settings_dialog(dlg, self.tr) 
        if dlg.exec() == QDialog.DialogCode.Accepted: 
            settings = dlg.get_settings()
            print(f"[DEBUG] Settings retornadas do diálogo: {settings}")
            idioma_antigo = self.idioma
            tema_antigo = self.tema
            self.cache_format = settings['cache_format']
            self.cache_quality = settings['cache_quality']
            self.num_threads = settings['num_threads']
            print(f"[DEBUG] self.num_threads APÓS set: {self.num_threads}")
            self.vram_limit = settings['vram_limit']
            self.device = settings['device']
            self.idioma = settings['idioma']
            self.tema = settings['tema']
            print(f"[DEBUG] self.idioma APÓS set: {self.idioma}, self.tema APÓS set: {self.tema}")
            self.loglevel = settings['loglevel']
            self.save_logs = settings['save_logs']
            self.output_dir = settings['output_dir']
            if hasattr(self, 'video_processor'):
                self.video_processor.DEFAULT_CACHE_FORMAT = self.cache_format
                self.video_processor.DEFAULT_CACHE_QUALITY = self.cache_quality
                if self.num_threads != self.video_processor.num_threads:
                    self.video_processor.num_threads = self.num_threads
                    print(f"[Config] Threads atualizado para: {self.num_threads}")
                self.video_processor.output_dir = self.output_dir
            self.save_config() 
            tema_map_inv = {'Claro': 'Light', 'Escuro': 'Dark', 'Светлая': 'Light', 'Тёмная': 'Dark', '浅色': 'Light', '深色': 'Dark'}
            tema_novo_norm = tema_map_inv.get(self.tema, 'Light')
            tema_antigo_norm = tema_map_inv.get(tema_antigo, 'Light')
            print(f"[DEBUG] Chamando apply_theme com tema: {self.tema}, Normalizado: {tema_novo_norm}")
            if tema_novo_norm != tema_antigo_norm:
                apply_theme(QtWidgets.QApplication.instance(), self.tema) 
            print(f"[DEBUG] Chamando apply_language com idioma: {self.idioma}")
            if self.idioma != idioma_antigo:
                self.apply_language(self.idioma) 

    def get_current_settings_for_dialog(self):
        """ Método auxiliar para obter as configurações atuais da MainWindow. """
        return {
            'idioma': getattr(self, 'idioma', 'Português'),
            'tema': getattr(self, 'tema', 'Claro'),
            'device': getattr(self, 'device', 'Auto'),
            'num_threads': getattr(self, 'num_threads', 2),
            'vram_limit': getattr(self, 'vram_limit', 4096),
            'output_dir': getattr(self, 'output_dir', str(Path.home())),
            'cache_format': getattr(self, 'cache_format', 'webp'),
            'cache_quality': getattr(self, 'cache_quality', 90),
            'loglevel': getattr(self, 'loglevel', 'Informativo'),
            'save_logs': getattr(self, 'save_logs', False),
        }

# (Opcional) Código para iniciar a aplicação, se este for o script principal:
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     # Aplica atributos de escala antes de criar a janela principal
#     if hasattr(QtCore.Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
#         QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
#     if hasattr(QtCore.Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
#         QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())