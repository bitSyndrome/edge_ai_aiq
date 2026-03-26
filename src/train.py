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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Air Quality Model Training")
    parser.add_argument("--model", default="AirQualityCNN",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture (default: AirQualityCNN)")
    parser.add_argument("--rawdata", default="rawdata", help="CSV directory path")
    parser.add_argument("--window", type=int, default=30, help="Time window size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="models", help="Model save directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 준비
    train_ds, val_ds = create_datasets(args.rawdata, window_size=args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    # 모델 생성
    num_features = 5
    num_classes = 4
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(num_features, args.window, num_classes).to(device)
    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 학습 루프
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_acc = 0
    best_epoch = 0
    history = []

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:9.1%} | {val_loss:10.4f} {val_acc:9.1%}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = val_loss
            model_tag = args.model.replace("AirQuality", "").lower()
            save_path = os.path.join(args.output, f"air_quality_{model_tag}_{timestamp}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "window_size": args.window,
                "num_features": num_features,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, save_path)
            print("  *best*")
        else:
            print()

    # 클래스별 정확도 계산
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

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1%}")
    print(f"Model saved: {save_path}")

    # 리포트 생성
    report_path = os.path.join(args.output, f"report_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(f"# Training Report\n\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Model file: `{os.path.basename(save_path)}`\n\n")

        f.write(f"## Hyperparameters\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| Window size | {args.window} |\n")
        f.write(f"| Batch size | {args.batch} |\n")
        f.write(f"| Learning rate | {args.lr} |\n")
        f.write(f"| Epochs | {args.epochs} |\n")
        f.write(f"| Optimizer | Adam |\n")
        f.write(f"| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |\n")
        f.write(f"| Model parameters | {sum(p.numel() for p in model.parameters()):,} |\n\n")

        f.write(f"## Dataset\n\n")
        f.write(f"| Split | Samples |\n|---|---|\n")
        f.write(f"| Train | {len(train_ds):,} |\n")
        f.write(f"| Validation | {len(val_ds):,} |\n\n")

        f.write(f"## Labeling Criteria\n\n")
        f.write(f"| Sensor | Good | Normal | Bad | Very Bad |\n")
        f.write(f"|---|---|---|---|---|\n")
        for sensor, th in QUALITY_THRESHOLDS.items():
            f.write(f"| {sensor} | ~{th[0]} | {th[0]+1}~{th[1]} | {th[1]+1}~{th[2]} | {th[2]+1}~ |\n")
        f.write(f"\n")

        f.write(f"## Best Result (Epoch {best_epoch})\n\n")
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
        f.write(f"| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |\n")
        f.write(f"|---|---|---|---|---|\n")
        for ep, tl, ta, vl, va in history:
            marker = " **best**" if ep == best_epoch else ""
            f.write(f"| {ep} | {tl:.4f} | {ta:.1%} | {vl:.4f} | {va:.1%}{marker} |\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
