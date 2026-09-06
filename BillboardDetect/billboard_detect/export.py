from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from billboard_detect.detectors import Detection


@dataclass
class BoxObs:
    frame: int
    t: float
    xyxy: tuple[float, float, float, float]
    conf: float
    label: str
    interpolated: bool = False


@dataclass
class TrackBook:
    observations: dict[int, list[BoxObs]] = field(default_factory=lambda: defaultdict(list))

    def add(
        self,
        track_id: int,
        *,
        frame: int,
        t: float,
        xyxy: tuple[float, float, float, float],
        conf: float,
        label: str,
        interpolated: bool = False,
    ) -> None:
        self.observations[int(track_id)].append(
            BoxObs(
                frame=frame,
                t=t,
                xyxy=xyxy,
                conf=conf,
                label=label,
                interpolated=interpolated,
            )
        )

    def by_frame(self) -> dict[int, list[tuple[int, BoxObs]]]:
        indexed: dict[int, list[tuple[int, BoxObs]]] = defaultdict(list)
        for track_id, boxes in self.observations.items():
            for box in boxes:
                indexed[box.frame].append((int(track_id), box))
        return indexed

    def summaries(self) -> list[dict[str, Any]]:
        tracks = []
        for track_id in sorted(self.observations):
            boxes = self.observations[track_id]
            if not boxes:
                continue
            labels = [box.label for box in boxes]
            label = max(set(labels), key=labels.count)
            real = [box for box in boxes if not box.interpolated]
            confs = [box.conf for box in (real or boxes)]
            tracks.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "first_frame": boxes[0].frame,
                    "last_frame": boxes[-1].frame,
                    "t_start": round(boxes[0].t, 3),
                    "t_end": round(boxes[-1].t, 3),
                    "duration_s": round(max(0.0, boxes[-1].t - boxes[0].t), 3),
                    "avg_confidence": round(sum(confs) / len(confs), 4),
                    "max_confidence": round(max(confs), 4),
                    "n_detections": len(real),
                    "n_interpolated": len(boxes) - len(real),
                    "boxes": [
                        {
                            "frame": box.frame,
                            "t": round(box.t, 3),
                            "xyxy": [round(v, 1) for v in box.xyxy],
                            "conf": round(box.conf, 4),
                            "label": box.label,
                            **({"interpolated": True} if box.interpolated else {}),
                        }
                        for box in boxes
                    ],
                }
            )
        return tracks


def write_tracks_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_raw_detections(
    path: Path,
    *,
    source: str,
    fps: float,
    width: int,
    height: int,
    detector: str,
    classes: list[str],
    detect_conf: float,
    frame_stride: int,
    n_frames: int,
    by_frame: dict[int, list[Detection]],
) -> None:
    frames = []
    for index in sorted(by_frame):
        frames.append(
            {
                "frame": index,
                "t": round(index / fps, 3) if fps else 0.0,
                "boxes": [
                    {
                        "xyxy": [round(v, 1) for v in det.xyxy],
                        "conf": round(det.confidence, 4),
                        "label": det.label,
                    }
                    for det in by_frame[index]
                ],
            }
        )
    write_tracks_json(
        path,
        {
            "source": source,
            "fps": fps,
            "width": width,
            "height": height,
            "detector": detector,
            "classes": classes,
            "detect_conf": detect_conf,
            "frame_stride": frame_stride,
            "n_frames": n_frames,
            "frames": frames,
        },
    )


def load_tracks_json(path: Path) -> tuple[dict[str, Any], TrackBook]:
    payload = json.loads(path.read_text())
    book = TrackBook()
    for track in payload.get("tracks") or []:
        track_id = int(track["track_id"])
        label = str(track.get("label") or "")
        for box in track.get("boxes") or []:
            xyxy = box["xyxy"]
            book.add(
                track_id,
                frame=int(box["frame"]),
                t=float(box["t"]),
                xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                conf=float(box.get("conf") or 0.0),
                label=str(box.get("label") or label),
                interpolated=bool(box.get("interpolated", False)),
            )
    return payload, book


def load_raw_detections(path: Path) -> tuple[dict[str, Any], dict[int, list[Detection]]]:
    payload = json.loads(path.read_text())
    by_frame: dict[int, list[Detection]] = {}
    for entry in payload.get("frames", []):
        boxes = []
        for box in entry.get("boxes", []):
            xyxy = box["xyxy"]
            boxes.append(
                Detection(
                    xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(box["conf"]),
                    label=str(box["label"]),
                )
            )
        by_frame[int(entry["frame"])] = boxes
    return payload, by_frame


def filter_detections(by_frame: dict[int, list[Detection]], conf: float) -> dict[int, list[Detection]]:
    return {
        index: [det for det in dets if det.confidence >= conf]
        for index, dets in by_frame.items()
    }


def nms_detections(
    dets: list[Detection],
    iou_threshold: float = 0.5,
    *,
    max_area_frac: float = 0.20,
    frame_area: float | None = None,
) -> list[Detection]:
    """Keep the highest-confidence box when several overlap. Never union boxes."""
    filtered = list(dets)
    if frame_area and frame_area > 0 and max_area_frac > 0:
        cap = max_area_frac * frame_area
        filtered = [det for det in filtered if _area(det.xyxy) <= cap]
    if len(filtered) <= 1:
        return filtered
    remaining = sorted(filtered, key=lambda det: det.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [det for det in remaining if _iou(best.xyxy, det.xyxy) < iou_threshold]
    return kept


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _area(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def write_appearances_csv(path: Path, tracks: list[dict[str, Any]]) -> None:
    fields = [
        "track_id",
        "label",
        "t_start",
        "t_end",
        "duration_s",
        "avg_confidence",
        "max_confidence",
        "n_detections",
        "n_interpolated",
        "first_frame",
        "last_frame",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            writer.writerow({key: track[key] for key in fields})
