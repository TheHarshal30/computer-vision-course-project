from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - keeps the module usable without torch installed
    class Dataset:  # type: ignore[override]
        pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image(path: Path) -> "Image.Image":
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("Pillow is required to load images. Install it with `pip install pillow`.") from exc

    return Image.open(path).convert("RGB")


def obb_to_xyxy(points: Iterable[float]) -> list[float]:
    coords = list(points)
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def read_yolo_obb_labels(label_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not label_path.exists():
        return records

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 9:
                continue

            class_id = int(parts[0])
            obb = [float(value) for value in parts[1:]]
            records.append(
                {
                    "class_id": class_id,
                    "obb": obb,
                    "xyxy": obb_to_xyxy(obb),
                }
            )

    return records


@dataclass(frozen=True)
class HelmetRecord:
    split: str
    label_name: str
    label_id: int
    subtype: str
    image_path: Path


class MotorcycleRiderDataset(Dataset):
    """PyTorch-style dataset for the processed motorcycle-rider OBB dataset."""

    def __init__(
        self,
        root: str | Path = "datasets/processed/motorcycle_rider_obb",
        split: str = "train",
        image_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
        load_images: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.load_images = load_images
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        self.samples = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        metadata_path = self.root / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.samples[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        annotations = read_yolo_obb_labels(label_path)

        sample: dict[str, Any] = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "annotations": annotations,
            "classes": [record["class_id"] for record in annotations],
            "boxes_obb": [record["obb"] for record in annotations],
            "boxes_xyxy": [record["xyxy"] for record in annotations],
        }

        if self.load_images:
            image = _load_image(image_path)
            if self.image_transform is not None:
                image = self.image_transform(image)
            sample["image"] = image

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        return sample


class HelmetWearingDataset(Dataset):
    """PyTorch-style dataset for the processed helmet-wearing classification dataset."""

    def __init__(
        self,
        root: str | Path = "datasets/processed/helmet_wearing",
        split: str = "train",
        image_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        load_images: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.load_images = load_images

        manifest_path = self.root / f"{split}_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        self.records = self._read_manifest(manifest_path)

    def _read_manifest(self, manifest_path: Path) -> list[HelmetRecord]:
        records: list[HelmetRecord] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    HelmetRecord(
                        split=row["split"],
                        label_name=row["label_name"],
                        label_id=int(row["label_id"]),
                        subtype=row["subtype"],
                        image_path=self.root / row["image_relpath"],
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample: dict[str, Any] = {
            "image_path": str(record.image_path),
            "label_name": record.label_name,
            "label_id": record.label_id,
            "subtype": record.subtype,
        }

        if self.load_images:
            image = _load_image(record.image_path)
            if self.image_transform is not None:
                image = self.image_transform(image)
            sample["image"] = image

        label = record.label_id
        if self.target_transform is not None:
            label = self.target_transform(label)
        sample["label"] = label

        return sample


def build_default_datasets(
    root: str | Path = "datasets/processed",
    load_images: bool = True,
) -> dict[str, Dataset]:
    base = Path(root)
    return {
        "motorcycle_rider_train": MotorcycleRiderDataset(
            base / "motorcycle_rider_obb",
            split="train",
            load_images=load_images,
        ),
        "motorcycle_rider_val": MotorcycleRiderDataset(
            base / "motorcycle_rider_obb",
            split="val",
            load_images=load_images,
        ),
        "helmet_train": HelmetWearingDataset(
            base / "helmet_wearing",
            split="train",
            load_images=load_images,
        ),
        "helmet_val": HelmetWearingDataset(
            base / "helmet_wearing",
            split="val",
            load_images=load_images,
        ),
    }
