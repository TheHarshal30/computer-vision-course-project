from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the motorcycle/rider detector.")
    parser.add_argument(
        "--data",
        default="datasets/processed/motorcycle_rider_detect/data.yaml",
        help="Path to YOLO detection data.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/rider_detector")
    parser.add_argument("--name", default="yolov8n")
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
