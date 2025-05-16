@echo off
ECHO Analisando dados de perfilamento do VisoMaster...
ECHO.

set PYTHONPATH=%~dp0
cd /d %~dp0

IF EXIST dependencies\Python (
    ECHO Usando ambiente Python integrado...
    dependencies\Python\python.exe analyze_profile.py
) ELSE (
    ECHO Usando Python do sistema...
    python analyze_profile.py
)

ECHO.
ECHO Relatório de análise concluído.
ECHO.

pause 