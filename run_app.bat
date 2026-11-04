@echo off
rem Speak & Transcribe launcher for Windows. Double-click to run.
rem Extra arguments are passed to Streamlit, e.g.  run_app.bat --server.address 0.0.0.0
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] First run: creating a Python environment and installing packages.
    echo [setup] This downloads about 1 GB and can take 5-15 minutes.
    %PY% -m venv .venv || goto :nopython
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo.
echo Starting the app. Open http://localhost:8501 in Chrome or Edge.
echo Close this window (or press Ctrl+C) to stop it.
echo.
rem The app runs headless (see .streamlit\config.toml), so open the browser ourselves once it is up.
if "%~1"=="" start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run app.py %*
goto :eof

:nopython
echo.
echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" during setup), then run this file again.
pause
exit /b 1

:fail
echo.
echo Package installation failed. Check the internet connection, delete the .venv folder, and try again.
pause
exit /b 1
