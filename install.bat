@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Organic Analyzer - Installation

echo ========================================
echo   Organic Analyzer installation
echo ========================================
echo.

set "PYTHON_LAUNCHER="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.12"

if not defined PYTHON_LAUNCHER (
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.13"
)

if not defined PYTHON_LAUNCHER (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_LAUNCHER=python"
)

if not defined PYTHON_LAUNCHER (
    echo Python 3.12 or 3.13 was not found.
    echo Install 64-bit Python from https://www.python.org/downloads/
    echo and enable "Add Python to PATH", then run this file again.
    echo.
    echo Python 3.12 або 3.13 не знайдено.
    echo Установіть 64-бітний Python і ввімкніть "Add Python to PATH",
    echo після чого запустіть цей файл повторно.
    pause
    exit /b 1
)

%PYTHON_LAUNCHER% -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) else 1)"
if errorlevel 1 (
    echo Python 3.12 or 3.13 is required. / Потрібен Python 3.12 або 3.13.
    pause
    exit /b 1
)

%PYTHON_LAUNCHER% -c "import struct; raise SystemExit(0 if struct.calcsize('P') * 8 == 64 else 1)"
if errorlevel 1 (
    echo A 64-bit Python installation is required. / Потрібен 64-бітний Python.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo requirements.txt was not found. Download or copy the complete project.
    echo requirements.txt не знайдено. Завантажте або скопіюйте весь проєкт.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment... / Створення віртуального середовища...
    %PYTHON_LAUNCHER% -m venv ".venv"
    if errorlevel 1 goto install_error
) else (
    echo The virtual environment already exists. / Віртуальне середовище вже існує.
)

echo Updating pip... / Оновлення pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error

echo Installing packages... This can take several minutes.
echo Установлення пакетів... Це може тривати кілька хвилин.
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto install_error

if not exist "checkpoints\inaturalist" mkdir "checkpoints\inaturalist"
if not exist "checkpoints\bioscan" mkdir "checkpoints\bioscan"

echo.
echo Installation completed successfully.
echo Установлення успішно завершено.
echo.
echo Add the model checkpoints if they are not present, then run start_app.bat.
echo Додайте checkpoints моделей, якщо їх немає, і запустіть start_app.bat.
pause
exit /b 0

:install_error
echo.
echo Installation failed. Check the internet connection and free disk space.
echo Помилка встановлення. Перевірте інтернет-з'єднання та вільне місце на диску.
pause
exit /b 1
