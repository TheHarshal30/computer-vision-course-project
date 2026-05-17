from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
MOTORCYCLE_VAL_FRACTION = 0.15
HELMET_VAL_FRACTION = 0.15


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_items(items: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    val_items = sorted(shuffled[:val_count])
    train_items = sorted(shuffled[val_count:])
    return train_items, val_items


def obb_to_aabb(coords: list[float]) -> list[float]:
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return [x_center, y_center, width, height]


def convert_obb_label_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with src.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
            xc, yc, w, h = obb_to_aabb(coords)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_motorcycle_rider(
    raw_root: Path,
    output_root: Path,
    val_fraction: float = MOTORCYCLE_VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> None:
    source_images = raw_root / "train" / "images"
    source_labels = raw_root / "train" / "labels"
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Expected Roboflow YOLO export under {raw_root}")

    image_files = sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    train_images, val_images = split_items(image_files, val_fraction=val_fraction, seed=seed)

    ensure_clean_dir(output_root)
    for split, files in (("train", train_images), ("val", val_images)):
        for image_path in files:
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            copy_file(image_path, output_root / "images" / split / image_path.name)
            copy_file(label_path, output_root / "labels" / split / label_path.name)

    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: motorcycle",
            "  1: rider",
            "",
        ]
    )
    write_text(output_root / "data.yaml", yaml_text)

    metadata = {
        "dataset_name": "motorcycle_rider_obb",
        "label_format": "yolov8_obb",
        "source": str(raw_root.resolve()),
        "splits": {
            "train": len(train_images),
            "val": len(val_images),
        },
        "classes": {
            "0": "motorcycle",
            "1": "rider",
        },
        "seed": seed,
        "val_fraction": val_fraction,
    }
    write_json(output_root / "metadata.json", metadata)


def prepare_motorcycle_rider_detect(
    obb_root: Path,
    output_root: Path,
) -> None:
    ensure_clean_dir(output_root)

    for split in ("train", "val"):
        image_dir = obb_root / "images" / split
        label_dir = obb_root / "labels" / split
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = label_dir / f"{image_path.stem}.txt"
            copy_file(image_path, output_root / "images" / split / image_path.name)
            convert_obb_label_file(label_path, output_root / "labels" / split / label_path.name)

    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "nc: 2",
            "names:",
            "  0: motorcycle",
            "  1: rider",
            "",
        ]
    )
    write_text(output_root / "data.yaml", yaml_text)

    source_metadata = json.loads((obb_root / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "dataset_name": "motorcycle_rider_detect",
        "label_format": "yolo_detect_xywh",
        "source": str(obb_root.resolve()),
        "derived_from": "motorcycle_rider_obb",
        "splits": source_metadata["splits"],
        "classes": source_metadata["classes"],
    }
    write_json(output_root / "metadata.json", metadata)


def _helmet_samples_by_bucket(raw_root: Path) -> dict[tuple[str, str], list[Path]]:
    buckets: dict[tuple[str, str], list[Path]] = defaultdict(list)
    roots = {
        "correct_way": raw_root / "Correct way",
        "incorrect_way": raw_root / "Incorrect way",
    }
    for label_name, label_root in roots.items():
        if not label_root.exists():
            raise FileNotFoundError(f"Missing helmet subset: {label_root}")
        for subtype_dir in sorted(path for path in label_root.iterdir() if path.is_dir()):
            subtype = subtype_dir.name.replace(" ", "_").lower()
            for image_path in sorted(subtype_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    buckets[(label_name, subtype)].append(image_path)
    return buckets


def prepare_helmet_wearing(
    raw_root: Path,
    output_root: Path,
    val_fraction: float = HELMET_VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> None:
    buckets = _helmet_samples_by_bucket(raw_root)
    ensure_clean_dir(output_root)

    manifests: dict[str, list[dict[str, str | int]]] = {"train": [], "val": []}
    class_map = {"incorrect_way": 0, "correct_way": 1}
    split_counter: dict[str, int] = defaultdict(int)
    subtype_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (label_name, subtype), image_files in sorted(buckets.items()):
        train_images, val_images = split_items(image_files, val_fraction=val_fraction, seed=seed)
        for split, files in (("train", train_images), ("val", val_images)):
            for image_path in files:
                prefixed_name = f"{subtype}__{image_path.name}"
                dst = output_root / "images" / split / label_name / prefixed_name
                copy_file(image_path, dst)
                manifests[split].append(
                    {
                        "split": split,
                        "label_name": label_name,
                        "label_id": class_map[label_name],
                        "subtype": subtype,
                        "image_relpath": str(dst.relative_to(output_root)),
                    }
                )
                split_counter[split] += 1
                subtype_counter[label_name][subtype] += 1

    for split, rows in manifests.items():
        manifest_path = output_root / f"{split}_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "label_name", "label_id", "subtype", "image_relpath"],
            )
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "dataset_name": "helmet_wearing",
        "task": "classification",
        "source": str(raw_root.resolve()),
        "splits": dict(split_counter),
        "classes": class_map,
        "subtypes": {label: dict(counts) for label, counts in subtype_counter.items()},
        "seed": seed,
        "val_fraction": val_fraction,
    }
    write_json(output_root / "metadata.json", metadata)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_root = project_root / "datasets" / "raw"
    processed_root = project_root / "datasets" / "processed"

    prepare_motorcycle_rider(
        raw_root=raw_root / "motorcycle_rider",
        output_root=processed_root / "motorcycle_rider_obb",
    )
    prepare_motorcycle_rider_detect(
        obb_root=processed_root / "motorcycle_rider_obb",
        output_root=processed_root / "motorcycle_rider_detect",
    )
    prepare_helmet_wearing(
        raw_root=raw_root / "helmet_wearing",
        output_root=processed_root / "helmet_wearing",
    )

    print("Prepared datasets:")
    print(f"  - {processed_root / 'motorcycle_rider_obb'}")
    print(f"  - {processed_root / 'motorcycle_rider_detect'}")
    print(f"  - {processed_root / 'helmet_wearing'}")


if __name__ == "__main__":
    main()
