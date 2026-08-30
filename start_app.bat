@echo off
setlocal
cd /d "%~dp0"

set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" (
    echo Virtual environment not found.
    echo Run the installation commands from RUN_ON_ANOTHER_PC.md first.
    echo.
    echo Віртуальне середовище не знайдено.
    echo Спочатку виконайте команди встановлення з RUN_ON_ANOTHER_PC.md.
    pause
    exit /b 1
)

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
