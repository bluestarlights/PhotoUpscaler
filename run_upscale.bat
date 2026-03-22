@echo off
setlocal EnableExtensions

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set "INPUT=%~1"
set "OUTPUT=%~2"
shift
shift

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python executable not found in PATH.
  exit /b 1
)

python upscaler.py "%INPUT%" "%OUTPUT%" %*
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   run_upscale.bat input.mp4 output_4k.mp4 [additional options]
echo.
echo Examples:
echo   run_upscale.bat input.mp4 output_4k.mp4 --method lanczos
echo   run_upscale.bat input.mp4 output_4k_ai.mp4 --method realesrgan --realesrgan-bin realesrgan-ncnn-vulkan --realesrgan-model realesr-animevideov3
exit /b 1
