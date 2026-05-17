from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the helmet/no-helmet/motorcycle/license-plate detector.")
    parser.add_argument("--data", default="data/merged_dataset/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (lower for macOS)")
    parser.add_argument("--cache", default="disk", choices=["ram", "disk", "false"], help="Image cache strategy")
    parser.add_argument("--project", default="runs/violation_detector")
    parser.add_argument("--name", default="yolov8n")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Cosine LR scheduler (better for short training)")
    parser.add_argument("--close-mosaic", type=int, default=3, help="Disable mosaic in final N epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_val: bool | str
    if args.cache == "false":
        cache_val = False
    else:
        cache_val = args.cache

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
        cache=cache_val,
        workers=args.workers,
        amp=True,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        exist_ok=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
