import os
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_datasets, QUALITY_THRESHOLDS
from model import AirQualityCNN, AirQualityMLP, MODEL_REGISTRY

LABEL_NAMES = ["Good", "Normal", "Bad", "Very Bad"]


def train_one_epoch(model, loader, criterion, optimizer, device, task="classify"):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if task != "anomaly":
            correct += (output.argmax(1) == y).sum().item()

    avg_loss = total_loss / total
    if task == "anomaly":
        return avg_loss, None
    return avg_loss, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="classify"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if task != "anomaly":
            correct += (output.argmax(1) == y).sum().item()

    avg_loss = total_loss / total
    if task == "anomaly":
        return avg_loss, None
    return avg_loss, correct / total


def main():
    parser = argparse.ArgumentParser(description="Air Quality Model Training")
    parser.add_argument("--model", default="AirQualityCNN",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture (default: AirQualityCNN)")
    parser.add_argument("--task", default="classify",
                        choices=["classify", "forecast", "anomaly"],
                        help="Task type (default: classify)")
    parser.add_argument("--horizon", type=int, default=300,
                        help="Forecast horizon in steps (default: 300 = 5min)")
    parser.add_argument("--rawdata", default="rawdata", help="CSV directory path")
    parser.add_argument("--window", type=int, default=30, help="Time window size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="models", help="Model save directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task}")

    # 데이터 준비
    train_ds, val_ds = create_datasets(
        args.rawdata, window_size=args.window,
        task=args.task, horizon=args.horizon,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    # 모델 생성
    num_features = 5
    num_classes = 4
    ModelClass = MODEL_REGISTRY[args.model]

    if args.task == "anomaly":
        model = ModelClass(num_features=num_features, window_size=args.window).to(device)
        criterion = nn.MSELoss()
    else:
        model = ModelClass(num_features, args.window, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 학습 루프
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_metric = float("inf") if args.task == "anomaly" else 0
    best_epoch = 0
    history = []
    save_path = None

    if args.task == "anomaly":
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
        print("-" * 35)
    else:
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | {'Val Loss':>10} {'Val Acc':>10}")
        print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.task)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.task)
        scheduler.step(val_loss)

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        if args.task == "anomaly":
            print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f}", end="")
            is_best = val_loss < best_val_metric
        else:
            print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:9.1%} | {val_loss:10.4f} {val_acc:9.1%}", end="")
            is_best = val_acc > best_val_metric

        if is_best:
            if args.task == "anomaly":
                best_val_metric = val_loss
            else:
                best_val_metric = val_acc
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = val_loss
            best_val_acc = val_acc

            model_tag = args.model.replace("AirQuality", "").lower()
            task_tag = args.task if args.task != "classify" else ""
            name_parts = ["air_quality", model_tag]
            if task_tag:
                name_parts.append(task_tag)
            name_parts.append(timestamp)
            save_path = os.path.join(args.output, f"{'_'.join(name_parts)}.pth")

            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "task": args.task,
                "window_size": args.window,
                "num_features": num_features,
                "epoch": epoch,
            }
            if args.task == "anomaly":
                save_dict["val_loss"] = val_loss
            else:
                save_dict["num_classes"] = num_classes
                save_dict["val_acc"] = val_acc
            if args.task == "forecast":
                save_dict["horizon"] = args.horizon

            torch.save(save_dict, save_path)
            print("  *best*")
        else:
            print()

    if save_path is None:
        print("No model saved (no improvement during training)")
        return

    # 클래스별 정확도 계산 (분류 태스크만)
    if args.task != "anomaly":
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True)["model_state_dict"])
        model.eval()
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                for c in range(num_classes):
                    mask = y == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()

    if args.task == "anomaly":
        print(f"\nTraining complete. Best val loss: {best_val_metric:.6f}")
    else:
        print(f"\nTraining complete. Best val accuracy: {best_val_metric:.1%}")
    print(f"Model saved: {save_path}")

    # 리포트 생성
    report_path = os.path.join(args.output, f"report_{timestamp}.md")
    with open(report_path, "w") as f:
        task_label = {"classify": "Classification", "forecast": "Forecast", "anomaly": "Anomaly Detection"}
        f.write(f"# Training Report — {task_label[args.task]}\n\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Task: {args.task}\n")
        if args.task == "forecast":
            f.write(f"- Horizon: {args.horizon} steps ({args.horizon // 60} min)\n")
        f.write(f"- Model file: `{os.path.basename(save_path)}`\n\n")

        f.write(f"## Hyperparameters\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| Window size | {args.window} |\n")
        f.write(f"| Batch size | {args.batch} |\n")
        f.write(f"| Learning rate | {args.lr} |\n")
        f.write(f"| Epochs | {args.epochs} |\n")
        f.write(f"| Optimizer | Adam |\n")
        f.write(f"| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |\n")
        f.write(f"| Model parameters | {sum(p.numel() for p in model.parameters()):,} |\n")
        if args.task == "forecast":
            f.write(f"| Forecast horizon | {args.horizon} steps |\n")
        f.write(f"\n")

        f.write(f"## Dataset\n\n")
        f.write(f"| Split | Samples |\n|---|---|\n")
        f.write(f"| Train | {len(train_ds):,} |\n")
        f.write(f"| Validation | {len(val_ds):,} |\n\n")

        if args.task != "anomaly":
            f.write(f"## Labeling Criteria\n\n")
            f.write(f"| Sensor | Good | Normal | Bad | Very Bad |\n")
            f.write(f"|---|---|---|---|---|\n")
            for sensor, th in QUALITY_THRESHOLDS.items():
                f.write(f"| {sensor} | ~{th[0]} | {th[0]+1}~{th[1]} | {th[1]+1}~{th[2]} | {th[2]+1}~ |\n")
            f.write(f"\n")

        f.write(f"## Best Result (Epoch {best_epoch})\n\n")
        if args.task == "anomaly":
            f.write(f"| Metric | Train | Validation |\n|---|---|---|\n")
            f.write(f"| Loss (MSE) | {best_train_loss:.6f} | {best_val_loss:.6f} |\n\n")
        else:
            f.write(f"| Metric | Train | Validation |\n|---|---|---|\n")
            f.write(f"| Loss | {best_train_loss:.4f} | {best_val_loss:.4f} |\n")
            f.write(f"| Accuracy | {best_train_acc:.1%} | {best_val_acc:.1%} |\n\n")

            f.write(f"## Per-Class Accuracy (Validation)\n\n")
            f.write(f"| Class | Samples | Accuracy |\n|---|---|---|\n")
            for c in range(num_classes):
                acc = class_correct[c] / class_total[c] * 100 if class_total[c] > 0 else 0
                f.write(f"| {c} ({LABEL_NAMES[c]}) | {int(class_total[c]):,} | {acc:.1f}% |\n")
            f.write(f"\n")

        f.write(f"## Epoch History\n\n")
        if args.task == "anomaly":
            f.write(f"| Epoch | Train Loss | Val Loss |\n")
            f.write(f"|---|---|---|\n")
            for ep, tl, _, vl, _ in history:
                marker = " **best**" if ep == best_epoch else ""
                f.write(f"| {ep} | {tl:.6f} | {vl:.6f}{marker} |\n")
        else:
            f.write(f"| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |\n")
            f.write(f"|---|---|---|---|---|\n")
            for ep, tl, ta, vl, va in history:
                marker = " **best**" if ep == best_epoch else ""
                f.write(f"| {ep} | {tl:.4f} | {ta:.1%} | {vl:.4f} | {va:.1%}{marker} |\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
