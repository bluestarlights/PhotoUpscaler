# PhotoUpscaler (4K) - GPU 최적화

`basicsr/realesrgan/gfpgan` 의존성을 제거하고, **Swin2SR(Transformers)** 기반으로 동작합니다.
이번 수정에서는 **GPU 사용률/속도/품질 체감 개선**에 집중했습니다.

## 핵심 개선점
- 기본 모델을 `Swin2SR x4 (Real-World)`로 변경 (기존 x2 대비 품질 개선)
- CUDA 환경에서 `autocast(fp16)` 추론 적용
- UI에서 디바이스를 `auto/cuda/cpu`로 직접 선택 가능
- CPU fallback 시 진행 로그에 경고 표시

## 빠른 시작
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python app.py --inbrowser
```

브라우저: `http://127.0.0.1:7860`

## 권장 설정 (RTX 5090)
- 업스케일 모델: `Swin2SR x4 (Real-World, 권장)`
- 추론 디바이스: `cuda`
- 목표 긴 변: 기본 `3840`

## 사용 방법
1. `scan/` 폴더에 파일을 넣거나 UI에서 여러 파일 업로드
2. 모델/디바이스 선택 (`cuda` 권장)
3. `여러 사진 업스케일 시작` 클릭
4. 결과 미리보기 및 ZIP 다운로드

## 트러블슈팅
- UI 로그에 `device=cpu`가 뜨면 GPU를 못 쓰는 상태입니다.
  - NVIDIA 드라이버/CUDA/PyTorch CUDA 빌드를 확인하세요.
  - UI에서 디바이스를 `cuda`로 강제했는데 실패하면 CUDA 설정 문제입니다.
- 첫 실행은 모델 다운로드 때문에 시간이 걸릴 수 있습니다.
