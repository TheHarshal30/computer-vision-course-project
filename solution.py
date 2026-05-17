from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


VIOLATION_MODEL_CANDIDATES = [
    "violation_detector.pt",
    "traffic_violation_detector.pt",
    "best_violation.pt",
    "best.pt",
]
PERSON_MODEL_CANDIDATES = [
    "person_detector.pt",
    "yolov8n.pt",
]

# Indian plate formats: e.g. KA01AB1234, KA1AB1234, KA01A1234, DL12C1234
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$")

# Plates have a roughly horizontal aspect; reject implausible boxes before OCR
MIN_PLATE_ASPECT = 1.3
MAX_PLATE_ASPECT = 6.0

# Person-to-bike assignment thresholds
MIN_ASSIGNMENT_SCORE = 0.05


@dataclass
class Detection:
    cls_id: int
    conf: float
    xyxy: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class OptionalOCREngine:
    def __init__(self, model_storage_dir: Path, allow_download: bool = False) -> None:
        self.backend = None
        self.reader = None
        self.model_storage_dir = model_storage_dir
        self.use_gpu = torch.cuda.is_available()

        try:
            import easyocr  # type: ignore

            self.backend = "easyocr"
            self.reader = easyocr.Reader(
                ["en"],
                gpu=self.use_gpu,
                model_storage_directory=str(model_storage_dir),
                download_enabled=allow_download,
            )
        except Exception:
            self.backend = None
            self.reader = None

    def read_with_conf(self, image: np.ndarray) -> tuple[str, float]:
        """Return (normalized_text, avg_confidence). Empty string if unusable."""
        if self.backend != "easyocr" or self.reader is None or image.size == 0:
            return "", 0.0

        results = self.reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0

        text_parts: list[str] = []
        confs: list[float] = []
        for _, text, conf in results:
            text_parts.append(text)
            confs.append(float(conf))
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return normalize_plate_text("".join(text_parts)), avg_conf


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_format(text: str) -> bool:
    return bool(INDIAN_PLATE_REGEX.match(text))


def xyxy_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def point_in_box(point: tuple[float, float], box: np.ndarray) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def plate_aspect_ratio(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return w / h


def assignment_score(person: Detection, motorcycle: Detection) -> float:
    """Higher = better match. Combines IoU, vertical position, horizontal alignment.

    A real rider has their hip area inside or just above the motorcycle bbox,
    their feet near the bike's seat level (not at ground level), and their
    horizontal center aligned with the bike.
    """
    iou = xyxy_iou(person.xyxy, motorcycle.xyxy)

    px1, py1, px2, py2 = person.xyxy
    mx1, my1, mx2, my2 = motorcycle.xyxy
    person_cx = (px1 + px2) / 2.0
    person_bottom_y = py2
    bike_cx = (mx1 + mx2) / 2.0
    bike_height = max(1.0, my2 - my1)
    bike_width = max(1.0, mx2 - mx1)

    # Horizontal alignment: 1.0 when centers align, 0.0 when one bike-width apart
    horizontal_gap = abs(person_cx - bike_cx) / bike_width
    horizontal_score = max(0.0, 1.0 - horizontal_gap)

    # Feet-position: rider's feet should land near the seat level (upper-mid of bike)
    # not at ground level (below the bike bottom).
    # Optimal: person_bottom_y in [my1, my2]. Penalize if much below my2 (pedestrian behind).
    if person_bottom_y < my1:
        feet_score = 0.0  # person entirely above the bike
    elif person_bottom_y > my2 + 0.3 * bike_height:
        feet_score = 0.0  # feet on ground, well below bike — pedestrian
    else:
        # Closer to seat level (upper half of bike) is better
        seat_y = my1 + 0.4 * bike_height
        distance_from_seat = abs(person_bottom_y - seat_y) / bike_height
        feet_score = max(0.0, 1.0 - distance_from_seat)

    return 0.5 * iou + 0.3 * horizontal_score + 0.2 * feet_score


def assign_persons_to_motorcycles(
    persons: list[Detection], motorcycles: list[Detection]
) -> dict[int, list[Detection]]:
    """Hungarian-style assignment: each person joins at most one motorcycle.

    Returns dict mapping motorcycle index -> list of assigned persons.
    """
    assignments: dict[int, list[Detection]] = {i: [] for i in range(len(motorcycles))}
    if not persons or not motorcycles:
        return assignments

    # Build score matrix: rows = persons, cols = motorcycles
    n_persons = len(persons)
    n_bikes = len(motorcycles)

    score_matrix = np.zeros((n_persons, n_bikes), dtype=float)
    for pi, person in enumerate(persons):
        for bi, bike in enumerate(motorcycles):
            score_matrix[pi, bi] = assignment_score(person, bike)

    # Allow up to 3 riders per bike by tiling the bike columns
    max_per_bike = 3
    tiled = np.tile(score_matrix, (1, max_per_bike))
    cost = -tiled  # convert to cost (minimization)

    row_ind, col_ind = linear_sum_assignment(cost)
    for pi, ci in zip(row_ind, col_ind):
        bike_idx = ci % n_bikes
        if score_matrix[pi, bike_idx] >= MIN_ASSIGNMENT_SCORE:
            assignments[bike_idx].append(persons[pi])
    return assignments


def deskew_plate(crop: np.ndarray) -> np.ndarray:
    """Detect text orientation and rotate plate to be horizontal."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.0:
        return crop
    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def upscale_crop(crop: np.ndarray, factor: float = 2.0) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    return cv2.resize(crop, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)


def adaptive_threshold_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.violation_model = YOLO(str(self._resolve_weight(VIOLATION_MODEL_CANDIDATES)))
        self.person_model = YOLO(str(self._resolve_weight(PERSON_MODEL_CANDIDATES)))
        self.ocr_engine = OptionalOCREngine(self.model_dir / "easyocr", allow_download=False)

        self.violation_class_map = {
            "helmet": 0,
            "no_helmet": 1,
            "motorcycle": 2,
            "license_plate": 3,
        }
        self.person_class_id = 0

    def _resolve_weight(self, candidates: list[str]) -> Path:
        for name in candidates:
            path = self.model_dir / name
            if path.exists():
                return path
        for name in candidates:
            path = Path(name)
            if path.exists():
                return path
        raise FileNotFoundError(f"Missing model file in {self.model_dir}: tried {candidates}")

    def _predict_model(self, model: YOLO, image_path: str, conf: float = 0.25, imgsz: int = 960) -> list[Detection]:
        results = model.predict(image_path, verbose=False, conf=conf, imgsz=imgsz)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for cls_id, score, box in zip(classes, confs, xyxy):
                detections.append(Detection(cls_id=cls_id, conf=float(score), xyxy=box.astype(float)))
        return detections

    def _ocr_plate_ensemble(self, image: np.ndarray, plate: Detection) -> str:
        """Try several preprocessing variants, prefer regex-valid readings."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = clamp_box(plate.xyxy, width, height)
        if x2 <= x1 or y2 <= y1:
            return ""

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        variants = [
            crop,
            upscale_crop(crop, 2.0),
            deskew_plate(crop),
            adaptive_threshold_crop(crop),
        ]

        valid_candidates: list[tuple[str, float]] = []
        all_candidates: list[tuple[str, float]] = []

        for variant in variants:
            text, conf = self.ocr_engine.read_with_conf(variant)
            if not text:
                continue
            all_candidates.append((text, conf))
            if is_valid_plate_format(text):
                valid_candidates.append((text, conf))

        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            return valid_candidates[0][0]
        if all_candidates:
            # No regex-valid reading — only trust if confidence is high enough
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            best_text, best_conf = all_candidates[0]
            if best_conf >= 0.5 and 6 <= len(best_text) <= 12:
                return best_text
        return ""

    def _select_plate(self, motorcycle: Detection, plates: list[Detection]) -> Detection | None:
        """Pick the best plate for this motorcycle, filtering by aspect ratio sanity."""
        # Filter by aspect ratio first — discard nonsense detections
        sane_plates = [
            p for p in plates if MIN_PLATE_ASPECT <= plate_aspect_ratio(p.xyxy) <= MAX_PLATE_ASPECT
        ]
        if not sane_plates:
            return None
        # Prefer plates whose center is inside the motorcycle bbox
        inside = [p for p in sane_plates if point_in_box(p.center, motorcycle.xyxy)]
        if inside:
            return max(inside, key=lambda p: p.conf)
        # Fall back to nearest plate by IoU
        scored = [(xyxy_iou(p.xyxy, motorcycle.xyxy), p) for p in sane_plates]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
        return None

    def _count_helmet_violations(
        self,
        assigned_riders: list[Detection],
        helmets: list[Detection],
        no_helmets: list[Detection],
        motorcycle: Detection,
    ) -> tuple[int, int]:
        """Returns (rider_count, helmet_violations).

        Inverted-helmet logic: stronger helmet class is used to find absence.
        Fallback to no_helmet detections when no riders were assigned at all.
        """
        rider_count = len(assigned_riders)

        if assigned_riders:
            helmet_violations = 0
            for rider in assigned_riders:
                rx1, ry1, rx2, ry2 = rider.xyxy
                head_box = np.array([rx1, ry1, rx2, ry1 + 0.4 * (ry2 - ry1)])
                has_helmet = any(
                    xyxy_iou(h.xyxy, head_box) > 0.05 or point_in_box(h.center, head_box)
                    for h in helmets
                )
                if not has_helmet:
                    helmet_violations += 1
            return rider_count, helmet_violations

        # Person detector missed everyone — fall back to no_helmet class
        nearby_no_helmets = [nh for nh in no_helmets if point_in_box(nh.center, motorcycle.xyxy)]
        n = len(nearby_no_helmets)
        return n, n

    def predict(self, image_path: str) -> dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        violation_dets = self._predict_model(self.violation_model, image_path, conf=0.2)
        person_dets = self._predict_model(self.person_model, image_path, conf=0.15)

        motorcycles = [det for det in violation_dets if det.cls_id == self.violation_class_map["motorcycle"]]
        rider_people = [det for det in person_dets if det.cls_id == self.person_class_id]
        helmets = [det for det in violation_dets if det.cls_id == self.violation_class_map["helmet"]]
        no_helmets = [det for det in violation_dets if det.cls_id == self.violation_class_map["no_helmet"]]
        plates = [det for det in violation_dets if det.cls_id == self.violation_class_map["license_plate"]]

        # Global one-to-one rider assignment — fixes multi-bike double-counting
        bike_assignments = assign_persons_to_motorcycles(rider_people, motorcycles)

        violations: list[dict[str, Any]] = []
        for bike_idx, motorcycle in enumerate(motorcycles):
            assigned_riders = bike_assignments[bike_idx]
            rider_count, helmet_violations = self._count_helmet_violations(
                assigned_riders, helmets, no_helmets, motorcycle
            )

            if rider_count <= 2 and helmet_violations == 0:
                continue

            selected_plate = self._select_plate(motorcycle, plates)
            license_plate = ""
            if selected_plate is not None:
                license_plate = self._ocr_plate_ensemble(image, selected_plate)

            violations.append(
                {
                    "num_riders": rider_count,
                    "helmet_violations": helmet_violations,
                    "license_plate": license_plate,
                }
            )

        return {"violations": violations}
