# Edge AI - Air Quality Classification

RV1106 NPU 기반 Edge 디바이스를 위한 실내 공기질 분류 프로젝트입니다.

센서 데이터(온도, 습도, CO2, TVOC, PM2.5)를 경량 AI 모델로 분석하여 공기질을 4단계(좋음/보통/나쁨/매우나쁨)로 판정합니다.

## 주요 기능

- **경량 모델**: 1D-CNN (8,932 params) / MLP (27,716 params) 선택 가능
- **자동 라벨링**: 환경부 실내공기질 관리법 기준 4등급 분류
- **Edge 변환**: PyTorch → ONNX → RKNN 변환 파이프라인
- **대시보드**: Streamlit 기반 데이터 분석, 실시간 추론, 모델 구조 시각화

## 구조

```
src/
├── dataset.py       # 데이터셋 로드, 정규화, 라벨링
├── model.py         # 모델 정의 (AirQualityCNN, AirQualityMLP)
├── train.py         # 학습 파이프라인
├── export_onnx.py   # ONNX 변환
├── dashboard.py     # Streamlit 대시보드
└── view_model.py    # Netron 모델 뷰어
```

## 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Graphviz (모델 구조 시각화용)
sudo apt install graphviz
```

## 사용법

### 모델 학습

```bash
# CNN 모델 (기본)
python src/train.py

# MLP 모델
python src/train.py --model AirQualityMLP

# 옵션
python src/train.py --model AirQualityMLP --epochs 100 --lr 0.0005 --batch 128
```

### ONNX 변환

```bash
# 대화형 모델 선택
bash run_export_onnx.sh

# 직접 지정
python src/export_onnx.py --checkpoint models/air_quality_cnn_XXXX.pth
```

### 대시보드

```bash
bash run_dashboard.sh
```

| 탭 | 기능 |
|---|---|
| 데이터 분석 | CSV 센서 데이터 시각화, 통계 요약 |
| 실시간 추론 | 슬라이더로 센서값 입력 → 공기질 판정 |
| 모델 구조 (Netron) | 인터랙티브 모델 그래프 |
| 모델 구조 (Graphviz) | 네이티브 통합 그래프 |

## 모델 입출력

- **입력**: `[Batch, 30, 5]` — 30개 시계열 윈도우 x 5개 센서(temp, humi, co2, tvoc, pm2.5)
- **출력**: `[Batch, 4]` — 4등급 분류 확률 (좋음/보통/나쁨/매우나쁨)

## 라벨링 기준 (환경부 실내공기질 관리법)

| 센서 | 좋음 | 보통 | 나쁨 | 매우나쁨 |
|---|---|---|---|---|
| CO2 (ppm) | ~500 | 501~1000 | 1001~2000 | 2001~ |
| PM2.5 (ug/m³) | ~15 | 16~35 | 36~75 | 76~ |
| TVOC (ug/m³) | ~200 | 201~500 | 501~1000 | 1001~ |

## 개발 환경

- Python 3.10
- PyTorch 2.6.0+cu124
- ONNX Runtime
- Streamlit
- Target: Luckfox Pico (RV1106, NPU 0.5TOPS)

## License

MIT
