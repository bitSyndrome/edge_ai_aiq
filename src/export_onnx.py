import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np

from model import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", default="models/air_quality_best.pth", help="Model checkpoint path")
    parser.add_argument("--output", default="models/air_quality.onnx", help="ONNX output path")
    args = parser.parse_args()

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    window_size = ckpt["window_size"]
    num_features = ckpt["num_features"]
    task = ckpt.get("task", "classify")

    model_type = ckpt.get("model_type", "AirQualityCNN")
    ModelClass = MODEL_REGISTRY[model_type]

    if task == "anomaly":
        model = ModelClass(num_features=num_features, window_size=window_size)
    else:
        num_classes = ckpt["num_classes"]
        model = ModelClass(num_features, window_size, num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model: {model_type} | Task: {task}")
    if task == "forecast":
        print(f"Horizon: {ckpt.get('horizon', 'N/A')} steps")

    # ONNX Export
    dummy = torch.randn(1, window_size, num_features)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if task == "anomaly":
        output_names = ["reconstructed"]
    else:
        output_names = ["air_quality"]

    input_axes = {"sensor_input": {0: "batch"}}
    output_axes = {output_names[0]: {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["sensor_input"],
        output_names=output_names,
        dynamic_axes={**input_axes, **output_axes},
        opset_version=17,
    )
    print(f"ONNX exported: {args.output}")

    # 검증
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")

    # ONNX Runtime 추론 비교
    sess = ort.InferenceSession(args.output)
    x_np = dummy.numpy()

    with torch.no_grad():
        pt_out = model(dummy).numpy()
    ort_out = sess.run(None, {"sensor_input": x_np})[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"PyTorch vs ONNX max diff: {diff:.6e}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
