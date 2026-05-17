from __future__ import annotations

from pathlib import Path

import easyocr
import torch


def main() -> None:
    model_dir = Path("models/easyocr").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available(),
        model_storage_directory=str(model_dir),
        download_enabled=True,
    )
    print(f"EasyOCR models ready in: {model_dir}")


if __name__ == "__main__":
    main()
