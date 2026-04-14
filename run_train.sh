#!/bin/bash
source venv/bin/activate

quit_check() {
    if [ "$1" = "q" ] || [ "$1" = "Q" ]; then
        echo "종료합니다."
        exit 0
    fi
}

echo "========================================="
echo "  Air Quality Model Training  (q: 종료)"
echo "========================================="

# 태스크 선택
echo ""
echo "태스크 선택:"
echo "  1) classify  — 현재 공기질 판정"
echo "  2) forecast  — 5분 후 공기질 예측"
echo "  3) anomaly   — 센서 이상 탐지"
read -p "선택 [1-3] (기본: 1): " task_num
quit_check "$task_num"

case "${task_num:-1}" in
    1) TASK="classify" ;;
    2) TASK="forecast" ;;
    3) TASK="anomaly" ;;
    *) echo "잘못된 선택입니다."; exit 1 ;;
esac

# 모델 선택
if [ "$TASK" = "anomaly" ]; then
    MODEL="AirQualityAutoencoder"
    echo ""
    echo "모델: $MODEL (anomaly 전용)"
else
    echo ""
    echo "모델 선택:"
    echo "  1) AirQualityCNN"
    echo "  2) AirQualityMLP"
    read -p "선택 [1-2] (기본: 1): " model_num
    quit_check "$model_num"

    case "${model_num:-1}" in
        1) MODEL="AirQualityCNN" ;;
        2) MODEL="AirQualityMLP" ;;
        *) echo "잘못된 선택입니다."; exit 1 ;;
    esac
fi

# 에포크 설정
read -p "에포크 수 (기본: 50): " epochs
quit_check "$epochs"
EPOCHS="${epochs:-50}"

# 학습 실행
echo ""
echo "========================================="
echo "  Task:   $TASK"
echo "  Model:  $MODEL"
echo "  Epochs: $EPOCHS"
if [ "$TASK" = "forecast" ]; then
    echo "  Horizon: 300 (5분)"
fi
echo "========================================="
echo ""

CMD="python src/train.py --task $TASK --model $MODEL --epochs $EPOCHS --rawdata rawdata --window 30"
if [ "$TASK" = "forecast" ]; then
    CMD="$CMD --horizon 300"
fi

echo "실행: $CMD"
echo ""
$CMD
