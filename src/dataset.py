import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLS = ["temp", "humi", "co2", "tvoc", "pm2.5"]

# 정규화 범위 (센서 스펙 기반, 필요시 조정)
FEATURE_RANGES = {
    "temp":   (0, 50),
    "humi":   (0, 100),
    "co2":    (400, 5000),
    "tvoc":   (0, 2000),
    "pm2.5":  (0, 300),
}

# 공기질 등급 자동 라벨링 기준 (환경부 실내공기질 관리법 기준)
# 등급: 0=좋음, 1=보통, 2=나쁨, 3=매우나쁨
#
# CO2 (ppm)  - 실내공기질 유지기준 1000ppm
#   좋음: ~500, 보통: 501~1000, 나쁨: 1001~2000, 매우나쁨: 2001~
#
# PM2.5 (ug/m3) - 환경부 미세먼지 예보 등급 기준
#   좋음: 0~15, 보통: 16~35, 나쁨: 36~75, 매우나쁨: 76~
#
# TVOC (ug/m3) - 실내공기질 권고기준 500ug/m3 (다중이용시설)
#   좋음: 0~200, 보통: 201~500, 나쁨: 501~1000, 매우나쁨: 1001~
QUALITY_THRESHOLDS = {
    "co2":   [500, 1000, 2000],     # ppm  (환경부 실내공기질 유지기준)
    "pm2.5": [15, 35, 75],          # ug/m3 (환경부 미세먼지 예보등급)
    "tvoc":  [200, 500, 1000],      # ug/m3 (실내공기질 권고기준)
}


def auto_label(row):
    """환경부 권고 기준으로 공기질 등급 생성 (0=좋음, 1=보통, 2=나쁨, 3=매우나쁨).

    각 센서별 등급 중 가장 나쁜 등급을 최종 등급으로 채택.
    """
    worst = 0
    for col, thresholds in QUALITY_THRESHOLDS.items():
        val = row[col]
        if val > thresholds[2]:
            worst = max(worst, 3)
        elif val > thresholds[1]:
            worst = max(worst, 2)
        elif val > thresholds[0]:
            worst = max(worst, 1)
    return worst


def load_csv_files(rawdata_dir):
    """rawdata 디렉토리의 모든 CSV 파일을 하나의 DataFrame으로 로드."""
    csv_files = sorted(glob.glob(os.path.join(rawdata_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {rawdata_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(csv_files)} files")
    return combined


def preprocess(df):
    """결측치 처리, 정규화, 라벨 생성."""
    # 결측치 보간
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(method="linear").ffill().bfill()

    # 자동 라벨링 (정규화 전 원본 값으로)
    df["label"] = df.apply(auto_label, axis=1)

    # Min-Max 정규화 (0~1)
    for col in FEATURE_COLS:
        vmin, vmax = FEATURE_RANGES[col]
        df[col] = (df[col] - vmin) / (vmax - vmin)
        df[col] = df[col].clip(0, 1)

    return df


class SensorWindowDataset(Dataset):
    """시계열 윈도우 데이터셋. 과거 window_size개 샘플로 현재 공기질 등급 예측."""

    def __init__(self, features, labels, window_size=30):
        self.features = features  # (N, num_features)
        self.labels = labels      # (N,)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]  # (window, features)
        y = self.labels[idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def create_datasets(rawdata_dir, window_size=30, val_ratio=0.2):
    """CSV 로드 -> 전처리 -> 학습/검증 데이터셋 생성."""
    print("[1/3] Loading CSV files...")
    df = load_csv_files(rawdata_dir)

    print("[2/3] Preprocessing...")
    df = preprocess(df)

    features = df[FEATURE_COLS].values  # (N, 5)
    labels = df["label"].values         # (N,)

    # 라벨 분포 출력
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = {0: "Good", 1: "Normal", 2: "Bad", 3: "Very Bad"}[u]
        print(f"    Label {u} ({label_name}): {c} ({c/len(labels)*100:.1f}%)")

    # 시계열이므로 앞부분을 학습, 뒷부분을 검증으로 분할
    print("[3/3] Creating datasets...")
    total = len(features) - window_size
    split = int(total * (1 - val_ratio))

    train_ds = SensorWindowDataset(features[:split + window_size], labels[:split + window_size], window_size)
    val_ds = SensorWindowDataset(features[split:], labels[split:], window_size)

    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
