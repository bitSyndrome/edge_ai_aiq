#!/bin/bash
source venv/bin/activate
streamlit run src/dashboard.py --server.headless true --server.port 8501
