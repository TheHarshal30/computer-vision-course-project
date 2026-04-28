from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
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

    def read_text(self, image: np.ndarray) -> str:
        if self.backend != "easyocr" or self.reader is None or image.size == 0:
            return ""

        results = self.reader.readtext(image, detail=0, paragraph=False)
        return normalize_plate_text("".join(results))


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


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


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
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
            for cls_id, score, box in zip(classes, confs, xyxy, strict=True):
                detections.append(Detection(cls_id=cls_id, conf=float(score), xyxy=box.astype(float)))
        return detections

    def _assign_to_motorcycle(self, motorcycle: Detection, candidates: list[Detection]) -> list[Detection]:
        assigned: list[Detection] = []
        x1, y1, x2, y2 = motorcycle.xyxy
        width = x2 - x1
        height = y2 - y1
        expanded = np.array(
            [
                x1 - 0.35 * width,
                y1 - 0.10 * height,
                x2 + 0.35 * width,
                y2 + 0.10 * height,
            ],
            dtype=float,
        )
        for candidate in candidates:
            if point_in_box(candidate.center, expanded):
                assigned.append(candidate)
                continue
            overlap = xyxy_iou(candidate.xyxy, motorcycle.xyxy)
            horizontal_overlap = max(
                0.0,
                min(candidate.xyxy[2], motorcycle.xyxy[2]) - max(candidate.xyxy[0], motorcycle.xyxy[0]),
            )
            min_width = max(1.0, min(candidate.xyxy[2] - candidate.xyxy[0], width))
            overlap_ratio = horizontal_overlap / min_width
            if overlap > 0.01 or overlap_ratio > 0.35:
                assigned.append(candidate)
        return assigned

    def _ocr_plate(self, image: np.ndarray, plate: Detection) -> str:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = clamp_box(plate.xyxy, width, height)
        if x2 <= x1 or y2 <= y1:
            return ""

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        processed = preprocess_plate_crop(crop)
        text = self.ocr_engine.read_text(processed)
        return normalize_plate_text(text)

    def predict(self, image_path: str) -> dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        violation_dets = self._predict_model(self.violation_model, image_path, conf=0.2)
        person_dets = self._predict_model(self.person_model, image_path, conf=0.15)

        motorcycles = [det for det in violation_dets if det.cls_id == self.violation_class_map["motorcycle"]]
        rider_people = [det for det in person_dets if det.cls_id == self.person_class_id]
        no_helmets = [det for det in violation_dets if det.cls_id == self.violation_class_map["no_helmet"]]
        plates = [det for det in violation_dets if det.cls_id == self.violation_class_map["license_plate"]]

        violations: list[dict[str, Any]] = []
        for motorcycle in motorcycles:
            assigned_riders = self._assign_to_motorcycle(motorcycle, rider_people)
            rider_count = len(assigned_riders)

            assigned_no_helmets = []
            for no_helmet in no_helmets:
                if assigned_riders:
                    if any(xyxy_iou(no_helmet.xyxy, rider.xyxy) > 0.05 for rider in assigned_riders):
                        assigned_no_helmets.append(no_helmet)
                elif point_in_box(no_helmet.center, motorcycle.xyxy):
                    assigned_no_helmets.append(no_helmet)

            assigned_no_helmets = sorted(assigned_no_helmets, key=lambda det: det.conf, reverse=True)
            helmet_violations = len(assigned_no_helmets)

            if rider_count <= 2 and helmet_violations == 0:
                continue

            candidate_plates = [plate for plate in plates if point_in_box(plate.center, motorcycle.xyxy)]
            if not candidate_plates:
                candidate_plates = sorted(plates, key=lambda plate: xyxy_iou(plate.xyxy, motorcycle.xyxy), reverse=True)[:1]

            license_plate = ""
            if candidate_plates:
                license_plate = self._ocr_plate(image, candidate_plates[0])

            violations.append(
                {
                    "num_riders": rider_count,
                    "helmet_violations": helmet_violations,
                    "license_plate": license_plate,
                }
            )

        return {"violations": violations}
