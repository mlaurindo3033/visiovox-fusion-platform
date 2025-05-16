Resumo Minucioso do Projeto de Face Swap
1. Objetivo do Projeto
Construir um pipeline robusto de face swap, inspirado nos melhores projetos open source (FaceFusion, VisoMaster), utilizando:
MediaPipe para detecção e landmarks faciais.
ArcFace para extração de embeddings.
Inswapper128 (e outros modelos) para a troca de faces via ONNX.
Pipeline modular, extensível e com qualidade visual comparável aos líderes do segmento.
2. O que já foi implementado e validado
a) Alinhamento Facial
Implementação do alinhamento facial estilo ArcFace, usando os 5 pontos-chave corretos (olhos, nariz, cantos da boca).
Função _align_face_to_template validada com dados sintéticos e reais.
Testes visuais e unitários para garantir centralização e orientação corretas do crop facial.
b) Extração de Landmarks
Uso do MediaPipe FaceMesh para landmarks 468 pontos.
Fallback para landmarks 68 pontos (compatibilidade dlib).
Função _select_5_landmarks priorizando os índices corretos para ArcFace.
c) Embedding Facial
Pipeline para extração de embedding ArcFace, normalização L2, e adaptação para diferentes modelos.
Suporte a múltiplos modelos ArcFace (compatível com Inswapper, SimSwap, etc).
Testes para garantir que o embedding está correto e compatível com o modelo de swap.
d) Pipeline de Face Swap
Implementação do FaceSwapper:
Pré-processamento: conversão de cor, normalização, reshape, batch.
Chamada do modelo ONNX (Inswapper128) com input correto.
Pós-processamento: conversão de volta para BGR, uint8, reshape.
Colagem da face trocada na imagem original usando matriz inversa.
Blending com máscara oval e feathering.
Scripts de teste automatizados (test_full_face_swap.py) para validar o pipeline de ponta a ponta.
e) Debug e Diagnóstico
Salvamento de imagens intermediárias: crops alinhados, landmarks desenhados, saída do modelo, resultado final.
Logging detalhado em cada etapa do pipeline.
Scripts para extração automática de emap de modelos ONNX (mesmo que não encontrada).
f) Análise de Projetos de Referência
Download e análise dos códigos do FaceFusion e VisoMaster.
Comparação minuciosa dos pipelines, destacando diferenças em alinhamento, blending, enhancement, pós-processamento.
Inclusão dos projetos no workspace para consultas rápidas e comparações.
3. Situação Atual
O pipeline executa sem erros técnicos: todos os módulos, modelos e etapas funcionam.
O resultado visual do face swap está ruim: a face trocada aparece borrada, distorcida ou com artefatos, mesmo usando o mesmo modelo ONNX dos projetos de referência.
O mesmo modelo, nos projetos originais, gera resultados de alta qualidade.
Já foi descartado: problema de modelo corrompido, falta de emap, erro de alinhamento, erro de embedding, erro de blending básico.
4. O que já foi tentado para resolver
Validação visual e por código de todas as etapas do pipeline.
Testes com diferentes modelos ONNX (Inswapper128, SimSwap, etc).
Tentativa de extração de emap (não existe nos modelos baixados, igual aos projetos originais).
Análise detalhada dos pipelines do FaceFusion e VisoMaster para identificar possíveis diferenças ocultas.
Implementação de logs e debug em cada etapa.
5. Próximos Passos para Solução
Comparação Linha a Linha com Projetos Originais
Replicar exatamente o pré-processamento, normalização, ordem dos canais, dtype, etc. do FaceFusion/VisoMaster.
Garantir que o input do modelo ONNX seja idêntico ao dos projetos de referência.
Blending e Pós-processamento
Testar blending avançado (máscara customizada, feather, histogram matching, LAB).
Implementar enhancement pós-swap (CodeFormer, GFPGAN) se necessário.
Testes de Debug
Salvar e comparar crops, embeddings e outputs intermediários com os dos projetos originais.
Testar pipeline com imagens e embeddings idênticos aos usados nos projetos de referência.
Consulta a Comunidade/IA
Compartilhar o resumo do erro e exemplos de código/imagens com especialistas (ex: Gemini, fóruns, etc.) para identificar possíveis detalhes ocultos.
Aprimoramento do Pipeline
Incorporar processamento em lote, paralelismo com CUDA Streams, fallback automático e logs detalhados, como no visomaster_modificado.
6. O que falta para a solução definitiva
Identificar e corrigir qualquer diferença sutil no pipeline de pré-processamento, input/output do modelo ONNX, ou blending, que esteja causando a baixa qualidade visual.
Validar a solução com imagens reais, garantindo que o resultado final seja comparável ao dos projetos de referência.