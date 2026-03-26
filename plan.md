
## 📋 Edge AI 게이트웨이 구축 체크리스트

### 0단계: 데이터 수집 환경 작업
* [*] **BNS 거실 조절기 변경작업**: 각종 센서(온습도, 미세먼지, CO2, TVOC등) 데이터를 수집하기 위해 기존 세트 변경

### 1단계: 데이터 수집 및 규격화 (Data Preparation)
모델의 학습 기반이 되는 데이터의 '표준'을 만드는 단계입니다.
* [*] **센서 인터페이스 확인:** I2C/UART/ADC 등 각 센서의 통신 방식 및 드라이버 동작 확인.
* [*] **데이터 샘플링 주기 결정:** 공기질 변화를 감지하기 위한 최적의 주기 설정 : 1초.
* [*] **시계열 윈도우 크기 정의:** 추론에 사용할 과거 데이터의 개수 결정 : 30개의 데이터 샘플.
* [*] **정규화(Normalization) 기준 수립:** 각 센서의 Min/Max 값을 정의하고 $0 \sim 1$ 범위로 변환하는 수식 확정. (`src/dataset.py` FEATURE_RANGES)
* [*] **라벨링 기준 수립:** 환경부 실내공기질 관리법 기준으로 4등급(좋음/보통/나쁨/매우나쁨) 자동 라벨링 적용. (`src/dataset.py` QUALITY_THRESHOLDS)
* [*] **데이터셋 저장:** CSV 형태로 `rawdata/` 디렉토리에 날짜별 파일로 저장. 컬럼: timestamp, temp, humi, co2, tvoc, pm2.5

### 2단계: AI 모델 설계 및 학습 (Model Training)
RV1106 NPU 가속에 최적화된 모델을 생성하는 단계입니다.
* [*] **모델 구조 선택:** 1D-CNN 기반 경량 네트워크 (8,932 params, 36.6KB). (`src/model.py` AirQualityCNN)
* [*] **MLP 모델 추가:** MLP 기반 경량 네트워크. (`src/model.py` AirQualityMLP, `--model AirQualityMLP`으로 선택)
* [*] **입력 텐서 모양 고정:** `[Batch, 30, 5]` (Window=30, Features=temp/humi/co2/tvoc/pm2.5)
* [*] **모델 학습:** 분류(4등급) 학습 파이프라인 구축. 타임스탬프 포함 모델 저장 및 리포트 자동 생성. (`src/train.py`)
* [*] **ONNX 변환:** opset 17, dynamic batch axis 지원. 모델 선택 대화형 스크립트. (`src/export_onnx.py`, `run_export_onnx.sh`)
* [*] **모델 검증:** ONNX Runtime 추론 비교 완료 (PyTorch vs ONNX 오차 < 2e-6).
* [*] **모델 구조 시각화:** Netron(iframe) 및 Graphviz(네이티브) 두 가지 방식 대시보드 통합. 독립 뷰어도 지원. (`src/view_model.py`, `src/dashboard.py`)
* [*] **대시보드:** 데이터 분석, 실시간 추론, 모델 구조 시각화(Netron/Graphviz) 4개 탭. 사이드바에서 ONNX 모델 선택 공유. (`src/dashboard.py`)

### 3단계: RKNN 변환 및 최적화 (Model Conversion)
PC(Ubuntu)의 RKNN-Toolkit2를 사용하여 보드 전용 모델을 만듭니다.
* [ ] **RKNN-Toolkit2 환경 구축:** Docker 또는 가상환경에 변환 툴킷 설치.
* [ ] **Calibration 데이터 준비:** 양자화를 위해 실제 센서 데이터 중 일부(약 50~100개)를 별도 추출.
* [ ] **INT8 양자화 적용:** `do_quantization=True` 옵션으로 모델 용량 축소 및 NPU 최적화.
* [ ] **Target Platform 설정:** `target_platform='rv1106'`으로 지정하여 빌드.
* [ ] **RKNN 모델 반출:** 최종 생성된 `air_quality.rknn` 파일을 보드로 전송.

### 4단계: Luckfox RV1106 포팅 및 실행 (Deployment)
실제 보드에서 NPU를 구동하고 센서와 연동하는 단계입니다.
* [ ] **Luckfox SDK 설정:** 보드 사양에 맞는 C/C++ 교차 컴파일(Cross-compile) 환경 확인.
* [ ] **librknnrt 라이브러리 링크:** 보드 내 NPU 런타임 라이브러리 위치 확인 및 링크.
* [ ] **전처리 로직 C 구현:** 실시간 센서 데이터를 모델 입력 규격에 맞게 정규화하는 C 코드 작성.
* [ ] **NPU 추론 루프 구현:** `rknn_init` -> `rknn_inputs_set` -> `rknn_run` -> `rknn_outputs_get` 파이프라인 완성.
* [ ] **비동기 처리 설계:** 센서 수집(스레드 A)과 AI 추론(스레드 B)을 분리하여 실시간성 확보.
* [ ] **최종 결과 출력:** 추론된 등급을 LED, LCD 또는 상위 서버로 전송하는 로직 연동.


