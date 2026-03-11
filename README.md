# PhotoUpscaler (4K)

오래된 필름 스캔 사진(특히 70~80년대 인물 사진)을 **여러 장 한 번에** 4K로 업스케일하는 Gradio UI입니다.

## 핵심 기능
- 폴더 일괄 처리: `./scan` 폴더 재귀 검색
- 다중 업로드 처리: UI에서 여러 파일 직접 선택
- 배치 작업 결과 미리보기 + ZIP 다운로드
- 업스케일 엔진: **Real-ESRGAN x4+**
- 얼굴 복원(선택): **GFPGAN v1.4**

## 빠른 시작
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python app.py
```

브라우저: `http://localhost:7860`

## 사용 방법
1. `scan/` 폴더에 파일을 넣거나, UI에서 여러 이미지를 업로드합니다.
2. 출력 폴더와 해상도/타일 옵션을 지정합니다.
3. `여러 사진 업스케일 시작`을 누릅니다.
4. 결과 미리보기 확인 후 ZIP 파일을 다운로드합니다.

## RTX 5090 권장값
- 타일 기본: `512`
- 여유 VRAM: `1024`
- OOM 시: `256` 또는 `0`

## 비고
- 첫 실행 시 모델 가중치를 자동 다운로드합니다.
- CUDA가 없으면 CPU 모드로 동작하며 속도가 크게 느려집니다.
