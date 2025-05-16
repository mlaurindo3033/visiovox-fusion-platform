@echo off
ECHO Executando VisoMaster com perfilamento...
ECHO.

set PYTHONPATH=%~dp0
cd /d %~dp0

IF EXIST dependencies\Python (
    ECHO Usando ambiente Python integrado...
    dependencies\Python\python.exe main.py --profile --memory-monitor
) ELSE (
    ECHO Usando Python do sistema...
    python main.py --profile --memory-monitor
)

ECHO.
ECHO Resultados do perfilamento foram salvos em: profile_results/main_profile.txt
ECHO.

pause 