# PhotoUpscaler (4K Video Upscaler)

네이버 블로그 글(Comfy + RTX 가속 아이디어)을 바탕으로,
로컬에서 바로 돌려볼 수 있는 **4K 비디오 업스케일러 CLI**를 만들었습니다.

## 핵심 아이디어 반영

- 4K 업스케일 결과물 생성(기본 3840x2160)
- 빠른 기본 경로: `ffmpeg + lanczos`
- AI 업스케일 경로: `realesrgan-ncnn-vulkan` 연동(`--method realesrgan`)
- 향후 ComfyUI App View / TensorRT 파이프라인으로 확장 가능한 구조

## 요구사항

- Python 3.10+
- `ffmpeg`, `ffprobe`
- (선택) `realesrgan-ncnn-vulkan`

## 사용법

### 1) 기본 4K 업스케일 (빠르고 간단)

```bash
python upscaler.py input.mp4 output_4k.mp4 --method lanczos
```

### 2) Real-ESRGAN 기반 AI 업스케일

```bash
python upscaler.py input.mp4 output_4k_ai.mp4 \
  --method realesrgan \
  --realesrgan-bin realesrgan-ncnn-vulkan \
  --realesrgan-model realesr-animevideov3
```


### 3) Windows 배치파일로 실행

```bat
run_upscale.bat input.mp4 output_4k.mp4 --method lanczos
```

추가 옵션은 그대로 뒤에 붙이면 됩니다.


### 4) ComfyUI WebUI 실행 배치파일

ComfyUI는 말씀하신 것처럼 WebUI 기반입니다. 아래 배치파일을 추가했습니다.

```bat
start_comfyui_webui.bat
```

동작:
- `ComfyUI/main.py` 존재 여부 확인
- `http://127.0.0.1:8188` 브라우저 자동 오픈
- `python main.py --listen 127.0.0.1 --port 8188` 실행

필요하면 `start_comfyui_webui.bat` 상단의 `COMFYUI_DIR`, `PYTHON_EXE`, `HOST`, `PORT` 값을 환경에 맞게 수정하세요.

## 블로그 기반 권장 운영 팁

블로그에서 강조한 내용을 실제 운영 가이드로 정리하면:

1. NVIDIA 드라이버를 최신으로 유지(글에서는 551.76+ 언급)
2. ComfyUI / Stability Matrix 최신화
3. FP4/FP8/TensorRT 경로는 **생성 단계 가속**에 유리
4. 업스케일 단계는 RTX VSR 또는 AI 업스케일러 노드로 분리

이 저장소는 우선 "실행 가능한 최소 업스케일러"를 제공하고,
추후 ComfyUI 워크플로 JSON/노드 연동으로 확장하도록 설계했습니다.
