@echo off
setlocal EnableExtensions

REM ===== 사용자 환경에 맞게 수정 =====
set "COMFYUI_DIR=%~dp0ComfyUI"
set "PYTHON_EXE=python"
set "HOST=127.0.0.1"
set "PORT=8188"
REM ================================

if not exist "%COMFYUI_DIR%\main.py" (
  echo [ERROR] ComfyUI main.py not found: %COMFYUI_DIR%\main.py
  echo [HINT] set COMFYUI_DIR to your actual ComfyUI folder.
  exit /b 1
)

where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python executable not found: %PYTHON_EXE%
  exit /b 1
)

echo [INFO] Starting ComfyUI WebUI at http://%HOST%:%PORT%
start "" "http://%HOST%:%PORT%"

pushd "%COMFYUI_DIR%"
"%PYTHON_EXE%" main.py --listen %HOST% --port %PORT%
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
