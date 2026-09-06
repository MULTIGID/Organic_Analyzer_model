@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Organic Analyzer

set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" (
    echo Virtual environment not found.
    echo Run install.bat first.
    echo.
    echo Віртуальне середовище не знайдено.
    echo Спочатку запустіть install.bat.
    pause
    exit /b 1
)

if not exist "%~dp0app.py" (
    echo app.py was not found. Download or copy the complete project.
    echo app.py не знайдено. Завантажте або скопіюйте весь проєкт.
    pause
    exit /b 1
)

if not exist "%~dp0checkpoints\inaturalist\resnet50_inaturalist_best.pt" (
    echo Warning: the iNaturalist checkpoint is missing.
    echo Увага: checkpoint iNaturalist відсутній.
)
if not exist "%~dp0checkpoints\bioscan\bioscan_species_resnet50_best.pt" (
    echo Warning: the BIOSCAN checkpoint is missing.
    echo Увага: checkpoint BIOSCAN відсутній.
)
if not exist "%~dp0checkpoints\bioscan\classes.json" (
    echo Warning: the BIOSCAN class dictionary is missing.
    echo Увага: словник класів BIOSCAN відсутній.
)
echo.

echo Starting Biological Image Analyzer...
echo Open http://localhost:8501 if the browser does not open automatically.
echo Press Ctrl+C to stop the application.
echo.

"%APP_PYTHON%" -m streamlit run "%~dp0app.py" --server.address 0.0.0.0 --server.port 8501

if errorlevel 1 (
    echo.
    echo The application stopped with an error.
    pause
)

endlocal
