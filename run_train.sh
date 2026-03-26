#!/bin/bash
source venv/bin/activate
python src/train.py --rawdata rawdata --epochs 50 --window 30
