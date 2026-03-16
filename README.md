# PhotoUpscaler (4K) - Python 최신버전 호환

`basicsr/realesrgan/gfpgan` 의존성을 제거하고, **Swin2SR(Transformers)** 기반으로 변경했습니다.
이 구성은 Python 최신 버전(예: 3.12+)에서 상대적으로 설치 호환성이 좋습니다.

## 핵심 기능
- 폴더 일괄 처리 (`./scan` 재귀 검색)
- 다중 업로드 배치 처리
- 결과 미리보기 + ZIP 다운로드
- 업스케일 모델: `caidas/swin2SR-classical-sr-x2-64`

## 빠른 시작
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python app.py
```

브라우저: `http://127.0.0.1:7860`


## UI가 안 뜰 때
- 로컬 PC 실행 시: `python app.py --host 127.0.0.1 --port 7860 --inbrowser`
- 같은 PC 브라우저 주소: `http://127.0.0.1:7860`
- Docker/원격 서버면 `--host 0.0.0.0`로 실행하고 포트(7860) 포워딩이 필요합니다.
- 외부 접속 테스트가 필요하면 `--share` 옵션으로 임시 공개 링크를 만들 수 있습니다.

## 사용 방법
1. `scan/` 폴더에 이미지들을 넣거나 UI에서 여러 파일을 업로드
2. 목표 긴 변 해상도(기본 3840) 설정
3. `여러 사진 업스케일 시작` 클릭
4. 완료 후 ZIP 다운로드

## RTX 5090 팁
- CUDA가 감지되면 GPU에서 추론합니다.
- 첫 실행 시 Hugging Face 모델 다운로드가 필요합니다.
