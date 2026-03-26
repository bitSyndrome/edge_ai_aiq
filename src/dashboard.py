import os
import glob
import threading
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import onnxruntime as ort
import onnx
import graphviz
import netron

NETRON_PORT = 8081
_netron_lock = threading.Lock()

LABEL_NAMES = {0: "좋음", 1: "보통", 2: "나쁨", 3: "매우나쁨"}
LABEL_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

SENSORS = [
    {"col": "temp",  "name": "온도",   "unit": "°C"},
    {"col": "humi",  "name": "습도",   "unit": "%"},
    {"col": "co2",   "name": "CO2",    "unit": "ppm"},
    {"col": "tvoc",  "name": "TVOC",   "unit": "ug/m³"},
    {"col": "pm2_5", "name": "PM2.5",  "unit": "ug/m³"},
]
SENSOR_COLS = [s["col"] for s in SENSORS]

QUALITY_THRESHOLDS = {
    "co2":   [500, 1000, 2000],
    "pm2_5": [15, 35, 75],
    "tvoc":  [200, 500, 1000],
}

FEATURE_RANGES = {
    "temp":  (0, 50),
    "humi":  (0, 100),
    "co2":   (400, 5000),
    "tvoc":  (0, 2000),
    "pm2_5": (0, 300),
}

SLIDER_CONFIG = {
    "temp":  {"min": 0.0,   "max": 50.0,   "default": 24.0,  "step": 0.1},
    "humi":  {"min": 0.0,   "max": 100.0,  "default": 50.0,  "step": 0.1},
    "co2":   {"min": 400.0, "max": 5000.0, "default": 600.0, "step": 10.0},
    "tvoc":  {"min": 0.0,   "max": 2000.0, "default": 100.0, "step": 5.0},
    "pm2_5": {"min": 0.0,   "max": 300.0,  "default": 15.0,  "step": 0.5},
}

RAWDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "rawdata")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def auto_label(row):
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


@st.cache_data
def load_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(".", "_", regex=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label"] = df.apply(auto_label, axis=1)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df


@st.cache_resource
def load_onnx_model(model_path):
    sess = ort.InferenceSession(model_path)
    return sess


def normalize_value(col, val):
    vmin, vmax = FEATURE_RANGES[col]
    return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))


# ─── 데이터 분석 탭 ───

def page_data():
    csv_files = sorted(glob.glob(os.path.join(RAWDATA_DIR, "*.csv")))
    if not csv_files:
        st.error(f"rawdata 폴더에 CSV 파일이 없습니다: {RAWDATA_DIR}")
        return

    filenames = [os.path.basename(f) for f in csv_files]
    selected = st.sidebar.selectbox("CSV 파일 선택", filenames)
    df = load_csv(os.path.join(RAWDATA_DIR, selected))

    st.sidebar.subheader("시간 범위")
    min_time = df["timestamp"].min().to_pydatetime()
    max_time = df["timestamp"].max().to_pydatetime()
    time_range = st.sidebar.slider(
        "시간 범위 선택",
        min_value=min_time, max_value=max_time,
        value=(min_time, max_time), format="HH:mm:ss",
    )
    filtered = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]

    resample_options = {"1초 (원본)": None, "10초": "10s", "30초": "30s", "1분": "1min", "5분": "5min", "10분": "10min"}
    resample_label = st.sidebar.selectbox("리샘플링", list(resample_options.keys()), index=3)
    resample_rule = resample_options[resample_label]

    if resample_rule:
        plot_df = filtered.set_index("timestamp")[SENSOR_COLS].resample(resample_rule).mean().dropna().reset_index()
    else:
        plot_df = filtered

    # 메트릭 카드
    st.subheader(f"{selected} ({len(filtered):,} samples)")
    cols = st.columns(5)
    for i, sensor in enumerate(SENSORS):
        latest = filtered[sensor["col"]].iloc[-1] if len(filtered) > 0 else 0
        avg = filtered[sensor["col"]].mean()
        cols[i].metric(f"{sensor['name']} ({sensor['unit']})", f"{latest:.1f}", f"avg {avg:.1f}")

    col_chart, col_dist = st.columns([3, 1])

    with col_dist:
        st.subheader("등급 분포")
        label_counts = filtered["label_name"].value_counts()
        for label_id, name in LABEL_NAMES.items():
            count = label_counts.get(name, 0)
            pct = count / len(filtered) * 100 if len(filtered) > 0 else 0
            st.markdown(
                f'<span style="color:{LABEL_COLORS[label_id]}; font-size:1.2em;">●</span> '
                f'**{name}**: {count:,} ({pct:.1f}%)',
                unsafe_allow_html=True,
            )

        st.subheader("라벨링 기준")
        criteria_rows = [
            {"센서": "CO2 (ppm)",     "좋음": "~500",  "보통": "501~1000", "나쁨": "1001~2000", "매우나쁨": "2001~"},
            {"센서": "PM2.5 (ug/m³)", "좋음": "~15",   "보통": "16~35",    "나쁨": "36~75",     "매우나쁨": "76~"},
            {"센서": "TVOC (ug/m³)",  "좋음": "~200",  "보통": "201~500",  "나쁨": "501~1000",  "매우나쁨": "1001~"},
        ]
        st.dataframe(pd.DataFrame(criteria_rows), hide_index=True, use_container_width=True)

    with col_chart:
        st.subheader("센서 데이터 추이")
        for sensor in SENSORS:
            display_name = f"{sensor['name']} ({sensor['unit']})"
            chart_label = display_name.replace(".", "")
            st.caption(display_name)
            chart_data = plot_df.set_index("timestamp")[[sensor["col"]]].rename(columns={sensor["col"]: chart_label})
            st.line_chart(chart_data, height=200)

    st.subheader("통계 요약")
    stats = filtered[SENSOR_COLS].describe().T
    stats.index = [f"{s['name']} ({s['unit']})" for s in SENSORS]
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.columns = ["개수", "평균", "표준편차", "최소", "25%", "50%", "75%", "최대"]
    st.dataframe(stats.style.format({"개수": "{:.0f}", "평균": "{:.2f}", "표준편차": "{:.2f}",
                                      "최소": "{:.2f}", "25%": "{:.2f}", "50%": "{:.2f}",
                                      "75%": "{:.2f}", "최대": "{:.2f}"}),
                 use_container_width=True)


# ─── 실시간 추론 탭 ───

def page_inference():
    st.subheader("실시간 공기질 추론")

    selected_model = st.session_state.get("selected_onnx")
    if not selected_model:
        st.error("models/ 폴더에 ONNX 모델이 없습니다. 먼저 학습 및 ONNX 변환을 실행하세요.")
        return

    model_path = os.path.join(MODELS_DIR, selected_model)
    sess = load_onnx_model(model_path)

    st.caption(f"모델: {selected_model}")
    input_shape = sess.get_inputs()[0].shape
    window_size = input_shape[1]
    st.info(f"윈도우 크기: {window_size} — 슬라이더 값이 {window_size}개 동일하게 채워진 입력으로 추론합니다.")

    # 센서값 슬라이더
    st.markdown("### 센서 입력값")
    input_cols = st.columns(5)
    raw_values = {}
    for i, sensor in enumerate(SENSORS):
        cfg = SLIDER_CONFIG[sensor["col"]]
        raw_values[sensor["col"]] = input_cols[i].slider(
            f"{sensor['name']} ({sensor['unit']})",
            min_value=cfg["min"], max_value=cfg["max"],
            value=cfg["default"], step=cfg["step"],
        )

    # 정규화 후 윈도우 구성
    norm_values = [normalize_value(s["col"], raw_values[s["col"]]) for s in SENSORS]
    input_window = np.array([[norm_values] * window_size], dtype=np.float32)

    # 추론
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_window})[0]
    probs = np.exp(output[0]) / np.exp(output[0]).sum()
    pred_class = int(np.argmax(probs))

    # 결과 표시
    st.markdown("---")
    col_result, col_probs = st.columns([1, 2])

    with col_result:
        st.markdown("### 판정 결과")
        color = LABEL_COLORS[pred_class]
        st.markdown(
            f'<div style="text-align:center; padding:30px; border-radius:15px; '
            f'background-color:{color}20; border:3px solid {color};">'
            f'<span style="font-size:3em; color:{color};">{LABEL_NAMES[pred_class]}</span><br>'
            f'<span style="font-size:1.2em; color:gray;">신뢰도: {probs[pred_class]*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_probs:
        st.markdown("### 등급별 확률")
        for cls_id, name in LABEL_NAMES.items():
            prob = probs[cls_id]
            st.markdown(
                f'<span style="color:{LABEL_COLORS[cls_id]};">●</span> **{name}**',
                unsafe_allow_html=True,
            )
            st.progress(float(prob), text=f"{prob*100:.1f}%")

    # 규칙 기반 비교
    st.markdown("---")
    st.markdown("### 규칙 기반 비교 (환경부 기준)")
    rule_label = 0
    details = []
    for col, thresholds in QUALITY_THRESHOLDS.items():
        val = raw_values[col]
        if val > thresholds[2]:
            grade = 3
        elif val > thresholds[1]:
            grade = 2
        elif val > thresholds[0]:
            grade = 1
        else:
            grade = 0
        rule_label = max(rule_label, grade)
        sensor_name = next(s["name"] for s in SENSORS if s["col"] == col)
        details.append(f"{sensor_name}: **{LABEL_NAMES[grade]}**")

    compare_cols = st.columns(3)
    for i, detail in enumerate(details):
        compare_cols[i].markdown(detail)

    rule_color = LABEL_COLORS[rule_label]
    st.markdown(
        f'규칙 기반 종합 판정: <span style="color:{rule_color}; font-weight:bold; font-size:1.3em;">'
        f'{LABEL_NAMES[rule_label]}</span>',
        unsafe_allow_html=True,
    )


# ─── 모델 구조 탭 ───

def start_netron(model_path):
    import time, socket
    with _netron_lock:
        try:
            netron.stop()
        except Exception:
            pass
        for _ in range(10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", NETRON_PORT)) != 0:
                    break
            time.sleep(0.3)
        netron.start(model_path, address=("0.0.0.0", NETRON_PORT), browse=False)


def page_model_viewer():
    st.subheader("모델 구조 시각화")

    selected = st.session_state.get("selected_onnx")
    if not selected:
        st.error("models/ 폴더에 ONNX 모델이 없습니다.")
        return

    model_path = os.path.join(MODELS_DIR, selected)
    prev_loaded = st.session_state.get("netron_loaded")

    # 모델이 변경되었거나 아직 로드 안 된 경우 자동 시작
    if prev_loaded != selected:
        start_netron(model_path)
        st.session_state["netron_loaded"] = selected

    st.caption(f"모델: {selected}")
    # 타임스탬프로 iframe 캐시 무효화
    import time
    cache_bust = int(time.time())
    components.iframe(f"http://localhost:{NETRON_PORT}?v={cache_bust}", height=800, scrolling=True)


# ─── 모델 구조 (Graphviz) 탭 ───

NODE_COLORS = {
    "Conv": "#4a90d9",
    "Relu": "#e67e22",
    "Reshape": "#9b59b6",
    "Flatten": "#9b59b6",
    "MatMul": "#2ecc71",
    "Gemm": "#2ecc71",
    "Add": "#95a5a6",
    "AdaptiveAvgPool": "#1abc9c",
    "AveragePool": "#1abc9c",
    "GlobalAveragePool": "#1abc9c",
    "Transpose": "#f39c12",
    "Squeeze": "#f39c12",
    "Unsqueeze": "#f39c12",
}


def get_node_color(op_type):
    for key, color in NODE_COLORS.items():
        if key in op_type:
            return color
    return "#bdc3c7"


def get_shape_str(model, name):
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            shape = [d.dim_value if d.dim_value else "?" for d in dims]
            if shape:
                return f"[{','.join(str(s) for s in shape)}]"
    return ""


def build_model_graph(model_path):
    model = onnx.load(model_path)
    # shape 추론
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", bgcolor="transparent", fontname="Helvetica")
    dot.attr("node", fontname="Helvetica", fontsize="11", style="filled,rounded", shape="record")
    dot.attr("edge", fontname="Helvetica", fontsize="9", color="#666666")

    # 입력 노드
    for inp in model.graph.input:
        shape = get_shape_str(model, inp.name)
        dot.node(inp.name, label=f"Input\\n{inp.name}\\n{shape}",
                 fillcolor="#3498db", fontcolor="white")

    # 연산 노드
    for node in model.graph.node:
        node_id = node.output[0]
        color = get_node_color(node.op_type)
        shape = get_shape_str(model, node_id)
        label = f"{node.op_type}\\n{shape}" if shape else node.op_type
        dot.node(node_id, label=label, fillcolor=color, fontcolor="white")

        for inp_name in node.input:
            if inp_name:
                edge_label = get_shape_str(model, inp_name)
                dot.edge(inp_name, node_id, label=edge_label)

    # 출력 노드
    for out in model.graph.output:
        shape = get_shape_str(model, out.name)
        dot.node(f"out_{out.name}", label=f"Output\\n{out.name}\\n{shape}",
                 fillcolor="#e74c3c", fontcolor="white")
        dot.edge(out.name, f"out_{out.name}")

    return dot


def page_model_graph():
    st.subheader("모델 구조 (Graphviz)")

    selected = st.session_state.get("selected_onnx")
    if not selected:
        st.error("models/ 폴더에 ONNX 모델이 없습니다.")
        return

    model_path = os.path.join(MODELS_DIR, selected)
    st.caption(f"모델: {selected}")

    dot = build_model_graph(model_path)
    st.graphviz_chart(dot, use_container_width=True)

    # 모델 요약 정보
    model = onnx.load(model_path)
    st.markdown("### 모델 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("노드 수", len(model.graph.node))
    op_types = set(n.op_type for n in model.graph.node)
    col2.metric("연산 종류", len(op_types))
    col3.metric("파일 크기", f"{os.path.getsize(model_path) / 1024:.1f} KB")

    with st.expander("연산 목록"):
        for i, node in enumerate(model.graph.node):
            inputs = ", ".join(node.input)
            outputs = ", ".join(node.output)
            st.text(f"[{i}] {node.op_type}: {inputs} → {outputs}")


# ─── main ───

def main():
    st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
    st.title("Air Quality Sensor Dashboard")

    onnx_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    if onnx_files:
        onnx_names = [os.path.basename(f) for f in onnx_files]
        selected = st.sidebar.selectbox("ONNX 모델 선택", onnx_names, index=len(onnx_names) - 1)
        st.session_state["selected_onnx"] = selected

    tab_data, tab_inference, tab_viewer, tab_graph = st.tabs(
        ["📊 데이터 분석", "🤖 실시간 추론", "🔍 모델 구조 (Netron)", "📐 모델 구조 (Graphviz)"]
    )

    with tab_data:
        page_data()

    with tab_inference:
        page_inference()

    with tab_viewer:
        page_model_viewer()

    with tab_graph:
        page_model_graph()


if __name__ == "__main__":
    main()
