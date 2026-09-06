from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


DEFAULT_CLASSES = [
    "billboard",
    "digital billboard",
    "outdoor billboard",
    "LED billboard",
    "advertisement",
    "poster",
    "banner",
    "truck advertisement",
    "truck-side advertisement",
    "vehicle wrap",
]
DEFAULT_YOLO_WEIGHTS = "yolov8s-worldv2.pt"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str


class Detector(Protocol):
    name: str
    classes: list[str]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return boxes for a BGR uint8 frame."""


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_detector(
    name: str,
    *,
    classes: list[str],
    conf: float,
    device: str | None = None,
    weights: str | None = None,
    imgsz: int = 640,
) -> Detector:
    resolved = pick_device(device)
    if name == "yolo-world":
        return YOLOWorldDetector(
            classes=classes,
            conf=conf,
            device=resolved,
            weights=weights or DEFAULT_YOLO_WEIGHTS,
            imgsz=imgsz,
        )
    if name == "grounding-dino":
        return GroundingDinoDetector(
            classes=classes,
            conf=conf,
            device=resolved,
            model_id=weights or DEFAULT_DINO_MODEL,
        )
    raise ValueError(f"Unknown detector {name!r}. Use yolo-world or grounding-dino.")


class YOLOWorldDetector:
    name = "yolo-world"

    def __init__(
        self,
        classes: list[str],
        conf: float,
        device: str,
        weights: str = DEFAULT_YOLO_WEIGHTS,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLOWorld

        self.classes = list(classes)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.model = YOLOWorld(weights)
        self.model.set_classes(self.classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            iou=0.5,
            max_det=300,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names or {i: name for i, name in enumerate(self.classes)}
        out: list[Detection] = []
        for box in result.boxes:
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            fallback = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            if isinstance(names, dict):
                label = str(names.get(cls_id, fallback))
            else:
                label = str(names[cls_id]) if cls_id < len(names) else fallback
            out.append(
                Detection(
                    xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=float(box.conf[0]),
                    label=label,
                )
            )
        return out


class GroundingDinoDetector:
    name = "grounding-dino"

    def __init__(
        self,
        classes: list[str],
        conf: float,
        device: str,
        model_id: str = DEFAULT_DINO_MODEL,
    ) -> None:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise SystemExit(
                "Grounding DINO needs extra deps. From BillboardDetect run: pip install -e '.[dino]'"
            ) from exc

        self.classes = list(classes)
        self.conf = conf
        self.device = device
        self.prompt = " . ".join(c.strip().lower() for c in classes) + " ."
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = self._load_model(model_id)
        self.model.eval()

    def _load_model(self, model_id: str):
        from transformers import AutoModelForZeroShotObjectDetection

        try:
            return AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        except Exception:
            if self.device == "cpu":
                raise
            print(f"Grounding DINO failed on {self.device}; using CPU.")
            self.device = "cpu"
            return AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import torch
        from PIL import Image

        height, width = frame.shape[:2]
        image = Image.fromarray(frame[:, :, ::-1])
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(outputs, inputs, height, width)[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        raw_labels = results.get("text_labels")
        if raw_labels is None or (hasattr(raw_labels, "__len__") and len(raw_labels) == 0):
            raw_labels = results.get("labels", [])
        if boxes is None or len(boxes) == 0:
            return []
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        if hasattr(raw_labels, "cpu"):
            raw_labels = raw_labels.cpu().tolist()
        else:
            raw_labels = list(raw_labels)
        out: list[Detection] = []
        for i, (xyxy, score) in enumerate(zip(boxes, scores)):
            raw_label = raw_labels[i] if i < len(raw_labels) else self.classes[0]
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            out.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(score),
                    label=self._match_label(raw_label),
                )
            )
        return out

    def _post_process(self, outputs, inputs, height: int, width: int):
        kwargs = {
            "outputs": outputs,
            "input_ids": inputs["input_ids"],
            "text_threshold": self.conf,
            "target_sizes": [(height, width)],
        }
        try:
            return self.processor.post_process_grounded_object_detection(**kwargs, box_threshold=self.conf)
        except TypeError:
            return self.processor.post_process_grounded_object_detection(**kwargs, threshold=self.conf)

    def _match_label(self, raw: object) -> str:
        # Newer transformers return integer token ids in `labels`; those are not class indices.
        if isinstance(raw, (int, np.integer)):
            return self.classes[0] if self.classes else str(int(raw))
        text = str(raw).lower().strip().rstrip(".")
        for cls in self.classes:
            name = cls.lower()
            if name in text or text in name:
                return cls
        return text or (self.classes[0] if self.classes else "object")
