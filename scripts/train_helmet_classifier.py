from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a helmet-wearing classifier on the Mendeley dataset.")
    parser.add_argument("--data", default="datasets/processed/helmet_wearing/images", help="Classification dataset root")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Base classification checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--project", default="runs/helmet_classifier")
    parser.add_argument("--name", default="yolov8n_cls")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=True,
        cache=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
