# Edge AI Project

## 프로젝트 개요
Edge AI 개발 프로젝트입니다.

## 개발 환경
- OS: Linux (WSL2)
- Shell: bash
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM)
- CUDA: 12.4 (Driver: 12.7)
- Python: 3.10
- PyTorch: 2.6.0+cu124
- 가상환경: `venv/` (프로젝트 루트)

## 디렉토리 구조
```
_edge_ai/
├── CLAUDE.md
├── plan.md                 # 작업 체크리스트
├── rawdata/                # 센서 CSV 원본 데이터 (날짜별)
├── models/                 # 학습된 모델 (.pth) 및 ONNX (.onnx)
├── src/
│   ├── dataset.py          # 데이터셋 로드, 정규화, 라벨링
│   ├── model.py            # 모델 정의 (AirQualityCNN, AirQualityMLP)
│   ├── train.py            # 학습 파이프라인 (--model로 모델 선택)
│   ├── export_onnx.py      # ONNX 변환 (체크포인트에서 모델 타입 자동 감지)
│   ├── dashboard.py        # Streamlit 대시보드 (데이터분석/추론/모델구조)
│   └── view_model.py       # Netron 독립 모델 뷰어
├── run_train.sh            # 학습 실행 스크립트
├── run_export_onnx.sh      # ONNX 변환 스크립트 (모델 선택)
├── run_dashboard.sh        # 대시보드 실행 스크립트
└── notebooks/              # Jupyter 노트북
```

## 주요 명령어

### 환경 설정
```bash
# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# GPU 동작 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 모델 학습
```bash
# CNN 모델 학습 (기본값)
python src/train.py

# MLP 모델 학습
python src/train.py --model AirQualityMLP

# 또는 실행 스크립트 사용
bash run_train.sh
```

### 모델 변환 (Edge 배포용)
```bash
# ONNX 변환 (모델 선택 대화형)
bash run_export_onnx.sh

# 직접 지정
python src/export_onnx.py --checkpoint models/air_quality_cnn_XXXX.pth
```

### 대시보드
```bash
bash run_dashboard.sh
```
- 📊 데이터 분석: CSV 센서 데이터 시각화
- 🤖 실시간 추론: ONNX 모델로 공기질 추론
- 🔍 모델 구조 (Netron): iframe 기반 인터랙티브 뷰어
- 📐 모델 구조 (Graphviz): 네이티브 통합 그래프 뷰어

### 모델 뷰어 (독립 실행)
```bash
python src/view_model.py
```

### 테스트
```bash
pytest tests/
```

## 코드 작성 원칙
- 간결하고 명확한 코드 우선
- Edge 환경을 고려한 경량 모델 및 최적화 적용
- 불필요한 추상화 금지
- 외부 의존성 최소화

## 주의사항
- 대용량 모델 파일(.bin, .pt, .onnx)은 git에 커밋하지 않음
- `.env` 파일 및 인증 정보는 절대 커밋 금지
- 모델 가중치는 별도 저장소 또는 LFS 사용

## 작업계획
작업 체크리스트 : plan.md
