"""Merge five datasets into merged_dataset with unified class IDs.

Target classes:
  0 = helmet
  1 = no_helmet
  2 = motorcycle
  3 = license_plate

Source mappings (source_class_id -> target_class_id):
  helmet_lp     (cdio-zmfmj)      : 0 LP->3, 1 helmet->0, 2 no helmet->1
  motorcycle_lp (motorcycle-9gyny): 0 motorcycle->2, 1 new license plate->3
  tvd           (traffic-violation-detection): 0 No helmet->1; skip others
  with_no_helmet (traffic-violation): 0 Helmet->0, 1 motorcycle->2, 2 nohelmet->1, 3 platenumber->3
  indian_plates (holi-milan)       : 0 license_plate->3
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

# (source_dir_name, class_id_map)
SOURCES = [
    ("helmet_lp",      {0: 3, 1: 0, 2: 1}),
    ("motorcycle_lp",  {0: 2, 1: 3}),
    ("tvd",            {0: 1}),                          # only "No helmet"; skip rest
    ("with_no_helmet", {0: 0, 1: 2, 2: 1, 3: 3}),
    ("indian_plates",  {0: 3}),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_FRACTION = 0.15
SEED = 42


def remap_label_file(src: Path, dst: Path, cls_map: dict[int, int]) -> bool:
    lines = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        src_id = int(parts[0])
        if src_id not in cls_map:
            continue
        lines.append(f"{cls_map[src_id]} {' '.join(parts[1:])}")
    if not lines:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def collect_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def merge(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)

    here = Path(__file__).parent
    all_pairs: list[tuple[Path, Path, dict[int, int], str]] = []
    per_source_counts: dict[str, int] = {}

    for source_name, cls_map in SOURCES:
        source_root = here / "tmp" / source_name
        if not source_root.exists():
            print(f"  WARN: missing source {source_root}, skipping")
            continue
        pairs = collect_pairs(source_root)
        per_source_counts[source_name] = len(pairs)
        for img, lbl in pairs:
            all_pairs.append((img, lbl, cls_map, source_name))

    rng = random.Random(SEED)
    rng.shuffle(all_pairs)
    val_n = max(1, int(len(all_pairs) * VAL_FRACTION))
    val_set = all_pairs[:val_n]
    train_set = all_pairs[val_n:]

    def write_split(pairs, split_name):
        written = 0
        for idx, (img, lbl, cls_map, source_name) in enumerate(pairs):
            stem = f"{source_name}_{idx:06d}"
            dst_img = out / "images" / split_name / (stem + img.suffix)
            dst_lbl = out / "labels" / split_name / (stem + ".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if remap_label_file(lbl, dst_lbl, cls_map):
                shutil.copy2(img, dst_img)
                written += 1
        return written

    train_written = write_split(train_set, "train")
    val_written = write_split(val_set, "val")

    yaml = f"""path: {out.resolve()}
train: images/train
val: images/val

nc: 4
names:
  0: helmet
  1: no_helmet
  2: motorcycle
  3: license_plate
"""
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"Merged dataset written to: {out}")
    print(f"  train: {train_written} images")
    print(f"  val:   {val_written} images")
    print("Source contributions (before label-filtering):")
    for name, count in per_source_counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    merge(Path(__file__).parent / "merged_dataset")
