from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

MIN_DURATION_S = 0.05
MIN_REAL_BOXES = 2
MIN_SIDE_PX = 24
PAD_FRAC = 0.0
UPSCALE_SHORT_SIDE = 256
VLM_SHORT_SIDE = 512
KEYFRAMES = 1
KEYFRAME_SPACING_S = 0.3
CANDIDATE_SPACING_S = 0.1
CANDIDATE_MULTIPLIER = 1
MAX_CROP_AREA_FRAC = 0.02
VOTE_SIMILARITY = 0.75
SINGLE_READ_MIN_CONF = 0.5
DEFAULT_VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
OVERLAY_TEXT_MAX = 32
NOT_AD_TOKEN = "NOT_AD"
_NOT_AD_MARKERS = frozenset({"notad", "notbillboard", "notanad", "none", "nona", "na"})
VLM_SYSTEM_PROMPT = """You classify camera crops of outdoor objects.

Decide whether the object is an outdoor BILLBOARD ADVERTISEMENT: a large roadside or building-mounted commercial billboard, a digital billboard screen, a truck-side ad, or a vehicle wrap.

is_billboard is true for billboard ads AND for advertising painted or posted on trucks/vehicles (mobile billboards, vehicle wraps).

is_billboard is false for all of the following:
- street signs, traffic signs, wayfinding signs, street-name blades
- storefront signs, shop awnings, window lettering, door or facade logos
- brand logos on buildings that are not billboard ads
- traffic signals, trees, buildings, and random objects
- ordinary cars or trucks with no advertising graphic

Reply with ONE line of JSON and nothing else. Use this schema only:
{"is_billboard":true,"headline":"","full_copy":""}

Field rules:
- headline: brand name or main headline only (short). Empty if it is not a billboard.
- full_copy: all visible advertising copy on one line (spaces only, no newlines or markdown). It MUST include the headline: put the headline first, then any remaining copy. If the headline is the only readable text, full_copy equals headline.
- Copy ONLY words you can actually see in THIS image. Do not invent, complete, or recall a title, network, tagline, or date. Do not reuse copy from the instructions or from any other object.
- If it is a billboard but the copy is unreadable: {"is_billboard":true,"headline":"","full_copy":""}
- If it is NOT a billboard: {"is_billboard":false,"headline":"","full_copy":""}"""

_ENGINE: tuple[str, Any] | None = None


@dataclass(frozen=True)
class CropRequest:
    track_id: int
    frame: int
    t: float
    xyxy: tuple[float, float, float, float]
    padded_xyxy: tuple[float, float, float, float]
    conf: float
    label: str
    area: float


@dataclass
class OcrPlan:
    requests: list[CropRequest]
    by_frame: dict[int, list[CropRequest]] = field(default_factory=dict)
    tracks: dict[int, dict[str, Any]] = field(default_factory=dict)


def ensure_engine(engine: str = "rapidocr", *, model: str | None = None) -> None:
    _load_engine(engine, model=model)


def crop_short_side(engine: str) -> int:
    return VLM_SHORT_SIDE if engine == "vlm" else UPSCALE_SHORT_SIDE


def plan_ocr(
    tracks: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
    min_duration_s: float = MIN_DURATION_S,
) -> OcrPlan:
    """Pick real-detection crop candidates. Interpolated boxes are never used."""
    fps = fps if fps > 0 else 24.0
    by_frame: dict[int, list[CropRequest]] = defaultdict(list)
    requests: list[CropRequest] = []
    kept: dict[int, dict[str, Any]] = {}
    for track in tracks:
        duration = float(track.get("duration_s") or 0.0)
        real = [
            box
            for box in track.get("boxes") or []
            if not box.get("interpolated") and _min_side(box.get("xyxy") or []) >= MIN_SIDE_PX
        ]
        if duration < min_duration_s and len(real) < MIN_REAL_BOXES:
            continue
        if not real:
            continue
        track_id = int(track["track_id"])
        frame_area = max(1.0, float(width) * float(height))
        tight = [box for box in real if _area(box["xyxy"]) <= MAX_CROP_AREA_FRAC * frame_area]
        pool = tight or real
        early = sorted(pool, key=lambda box: int(box["frame"]))[:3]
        picked = [
            max(early, key=lambda box: float(box.get("conf") or 0.0) * max(_area(box["xyxy"]), 1.0))
        ]
        kept[track_id] = track
        for box in picked:
            xyxy = _as_xyxy(box["xyxy"])
            req = CropRequest(
                track_id=track_id,
                frame=int(box["frame"]),
                t=float(box["t"]),
                xyxy=xyxy,
                padded_xyxy=_pad_xyxy(xyxy, width, height, PAD_FRAC),
                conf=float(box.get("conf") or 0.0),
                label=str(box.get("label") or track.get("label") or ""),
                area=_area(xyxy),
            )
            requests.append(req)
            by_frame[req.frame].append(req)
    return OcrPlan(requests=requests, by_frame=dict(by_frame), tracks=kept)


def grab_crop(
    frame: np.ndarray,
    padded_xyxy: tuple[float, float, float, float],
    *,
    short_side: int = UPSCALE_SHORT_SIDE,
) -> tuple[np.ndarray, float] | None:
    """Return (upscaled BGR crop, native Laplacian sharpness) or None."""
    x1, y1, x2, y2 = (int(round(v)) for v in padded_xyxy)
    h, w = frame.shape[:2]
    x1 = min(max(x1, 0), w - 1)
    y1 = min(max(y1, 0), h - 1)
    x2 = min(max(x2, x1 + 1), w)
    y2 = min(max(y2, y1 + 1), h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    sharp = _sharpness(crop)
    return _upscale(crop, short_side), sharp


def run_ocr(
    plan: OcrPlan,
    crops: dict[tuple[int, int], tuple[np.ndarray, float]],
    *,
    source: str,
    tracks_file: str | None,
    fps: float,
    width: int,
    height: int,
    engine: str = "rapidocr",
    vlm_model: str | None = None,
) -> dict[str, Any]:
    fps = fps if fps > 0 else 24.0
    jobs = _select_jobs(plan, crops, fps)
    if engine == "vlm":
        track_rows = _run_vlm(plan, jobs, model=vlm_model)
    else:
        track_rows = _run_rapidocr(plan, jobs)
    return {
        "source": source,
        "tracks_file": tracks_file,
        "engine": engine,
        "vlm_model": vlm_model if engine == "vlm" else None,
        "min_duration_s": MIN_DURATION_S,
        "keyframes_per_track": KEYFRAMES,
        "keyframe_spacing_s": KEYFRAME_SPACING_S,
        "skip_interpolated": True,
        "width": width,
        "height": height,
        "fps": fps,
        "n_tracks": len(track_rows),
        "tracks": track_rows,
    }


def load_ocr_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def overlay_windows(payload: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Time span where a track's VLM headline is allowed on the overlay."""
    spans: dict[int, tuple[float, float]] = {}
    for track in payload.get("tracks") or []:
        if not is_ad_track(track):
            continue
        start = track.get("keep_t_start", track.get("t_start"))
        end = track.get("keep_t_end", track.get("t_end"))
        if start is None or end is None:
            continue
        spans[int(track["track_id"])] = (float(start), float(end))
    return spans


def overlay_labels(payload: dict[str, Any], *, max_chars: int = OVERLAY_TEXT_MAX) -> dict[int, str]:
    labels: dict[int, str] = {}
    for track in payload.get("tracks") or []:
        caption = overlay_caption(
            track.get("headline") or track.get("brand") or "",
            max_chars=max_chars,
        )
        if caption:
            labels[int(track["track_id"])] = caption
    return labels


def ad_track_ids(payload: dict[str, Any]) -> set[int]:
    return {int(track["track_id"]) for track in payload.get("tracks") or [] if is_ad_track(track)}


def apply_vlm_track_cuts(book: Any, payload: dict[str, Any], *, fps: float) -> int:
    """Keep a VLM label only on boxes that still match the classified crop.

    Leftover boxes (ID hijacks after the crop's object is gone) become new unlabeled tracks.
    """
    fps = fps if fps > 0 else 24.0
    rows = {int(track["track_id"]): track for track in payload.get("tracks") or []}
    next_id = max(book.observations, default=0) + 1
    n_split = 0
    n_trim = 0
    for track_id, boxes in list(book.observations.items()):
        row = rows.get(int(track_id))
        if row is None or not is_ad_track(row):
            continue
        crop = (row.get("keyframes") or [None])[0]
        if not crop:
            continue
        kept, rest = _partition_boxes_to_crop(boxes, crop, fps=fps)
        if not kept:
            continue
        if len(kept) != len(boxes):
            n_trim += 1
        book.observations[int(track_id)] = kept
        row["keep_t_start"] = round(kept[0].t, 3)
        row["keep_t_end"] = round(kept[-1].t, 3)
        row["t_start"] = row["keep_t_start"]
        row["t_end"] = row["keep_t_end"]
        row["duration_s"] = round(max(0.0, kept[-1].t - kept[0].t), 3)
        for segment in _contiguous_box_segments(rest, gap_s=0.4):
            book.observations[next_id] = segment
            next_id += 1
            n_split += 1
    payload["vlm_track_cuts"] = {"trimmed": n_trim, "split_fragments": n_split}
    return n_split


def is_ad_track(track: dict[str, Any]) -> bool:
    text = _flatten_copy(
        track.get("full_copy") or track.get("text") or track.get("headline") or track.get("brand") or ""
    )
    if _is_not_ad_text(text):
        return False
    if "is_ad" in track and track["is_ad"] is not None:
        return _truthy(track["is_ad"])
    return bool(text)


def overlay_caption(text: object, *, max_chars: int = OVERLAY_TEXT_MAX) -> str:
    caption = _flatten_copy(text, max_len=400)
    if _is_not_ad_text(caption):
        return ""
    if len(caption) <= max_chars:
        return caption
    clipped = caption[: max_chars - 1].rstrip(" ,;:-")
    if " " in clipped and len(clipped) > 8:
        clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,;:-")
    return clipped + "…"


def _include_headline(headline: object, full_copy: object) -> str:
    head = _flatten_copy(headline, max_len=80)
    copy = _flatten_copy(full_copy)
    if not head:
        return copy
    if not copy:
        return head
    if _normalize(head) and _normalize(head) in _normalize(copy):
        return copy
    return f"{head} {copy}".strip()


def write_ocr_outputs(output_dir: Path, stem: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}_ocr.json"
    csv_path = output_dir / f"{stem}_ocr.csv"
    for track in payload.get("tracks") or []:
        headline = _flatten_copy(track.get("headline") or track.get("brand") or "", max_len=80)
        full_copy = _include_headline(headline, track.get("full_copy") or track.get("text") or "")
        if _is_not_ad_text(full_copy) or _is_not_ad_text(headline):
            track["is_ad"] = False
            headline = ""
            full_copy = ""
        else:
            track["is_ad"] = is_ad_track({**track, "text": full_copy, "brand": headline})
        if not track["is_ad"]:
            headline = ""
            full_copy = ""
        track["headline"] = headline
        track["full_copy"] = full_copy
        track["brand"] = headline
        track["text"] = full_copy
        track.pop("raw", None)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = [
        "track_id",
        "label",
        "is_ad",
        "headline",
        "full_copy",
        "text_conf",
        "n_reads",
        "n_agree",
        "t_start",
        "t_end",
        "duration_s",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in payload.get("tracks") or []:
            row = {key: track.get(key, "") for key in fields}
            row["headline"] = _flatten_copy(row.get("headline") or "", max_len=80)
            row["full_copy"] = _flatten_copy(row.get("full_copy") or "")
            writer.writerow(row)
    return json_path, csv_path


def _select_jobs(
    plan: OcrPlan,
    crops: dict[tuple[int, int], tuple[np.ndarray, float]],
    fps: float,
) -> list[tuple[int, list[tuple[float, CropRequest, np.ndarray, float]]]]:
    by_track: dict[int, list[CropRequest]] = defaultdict(list)
    for req in plan.requests:
        if (req.track_id, req.frame) in crops:
            by_track[req.track_id].append(req)
    jobs: list[tuple[int, list[tuple[float, CropRequest, np.ndarray, float]]]] = []
    for track_id in sorted(by_track):
        scored: list[tuple[float, CropRequest, np.ndarray, float]] = []
        for req in by_track[track_id]:
            image, sharp = crops[(req.track_id, req.frame)]
            scored.append((req.area * req.conf * max(sharp, 1e-6), req, image, sharp))
        scored.sort(key=lambda row: row[1].t)
        jobs.append((track_id, scored[:1]))
    return jobs


def _run_rapidocr(
    plan: OcrPlan,
    jobs: list[tuple[int, list[tuple[float, CropRequest, np.ndarray, float]]]],
) -> list[dict[str, Any]]:
    engine = _load_engine("rapidocr")
    reads_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    flat = [(track_id, row) for track_id, rows in jobs for row in rows]
    for track_id, (_score, req, image, sharp) in tqdm(flat, desc="ocr", unit="crop"):
        lines = _read_crop(engine, image)
        text = " ".join(line["text"] for line in lines).strip()
        ocr_conf = float(sum(line["conf"] for line in lines) / len(lines)) if lines else 0.0
        reads_by_track[track_id].append(
            {
                "frame": req.frame,
                "t": round(req.t, 3),
                "xyxy": [round(v, 1) for v in req.xyxy],
                "padded_xyxy": [round(v, 1) for v in req.padded_xyxy],
                "conf": round(req.conf, 4),
                "sharpness": round(sharp, 2),
                "text": text,
                "ocr_conf": round(ocr_conf, 4),
                "lines": lines,
                "area": req.area,
            }
        )
    track_rows: list[dict[str, Any]] = []
    for track_id, _chosen in jobs:
        meta = plan.tracks[track_id]
        keyframes = []
        votes: list[tuple[str, float, float]] = []
        for item in reads_by_track[track_id]:
            area = float(item.pop("area"))
            keyframes.append(item)
            if item["text"]:
                votes.append((item["text"], float(item["ocr_conf"]), area))
        winner, win_conf, n_agree = _vote(votes)
        copy = _flatten_copy(winner)
        track_rows.append(
            {
                "track_id": track_id,
                "label": meta.get("label"),
                "t_start": meta.get("t_start"),
                "t_end": meta.get("t_end"),
                "duration_s": meta.get("duration_s"),
                "text": copy,
                "full_copy": copy,
                "brand": copy,
                "headline": copy,
                "is_ad": bool(copy),
                "text_conf": round(win_conf, 4),
                "n_reads": len(keyframes),
                "n_agree": n_agree,
                "keyframes": keyframes,
            }
        )
    return track_rows


def _run_vlm(
    plan: OcrPlan,
    jobs: list[tuple[int, list[tuple[float, CropRequest, np.ndarray, float]]]],
    *,
    model: str | None,
) -> list[dict[str, Any]]:
    vlm = _load_engine("vlm", model=model)
    flat: list[tuple[int, float, CropRequest, np.ndarray, float]] = []
    for track_id, chosen in jobs:
        if len(chosen) != 1:
            raise RuntimeError(f"VLM requires exactly one crop per track, got {len(chosen)} for track {track_id}")
        score, req, image, sharp = chosen[0]
        flat.append((track_id, score, req, image, sharp))
    per_crop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for track_id, _score, req, image, sharp in tqdm(flat, desc="vlm", unit="crop"):
        parsed = _read_crop_vlm(vlm, image)
        headline = _flatten_copy(parsed.get("headline") or parsed.get("brand") or "", max_len=80)
        full_copy = _include_headline(headline, parsed.get("full_copy") or parsed.get("text") or "")
        is_ad = _truthy(parsed.get("is_ad")) if parsed.get("is_ad") is not None else bool(full_copy)
        if _is_not_ad_text(full_copy) or _is_not_ad_text(headline):
            is_ad = False
        if not is_ad:
            headline = ""
            full_copy = ""
        per_crop[track_id].append(
            {
                "frame": req.frame,
                "t": round(req.t, 3),
                "xyxy": [round(v, 1) for v in req.xyxy],
                "padded_xyxy": [round(v, 1) for v in req.padded_xyxy],
                "conf": round(req.conf, 4),
                "sharpness": round(sharp, 2),
                "is_ad": is_ad,
                "headline": headline,
                "full_copy": full_copy,
            }
        )
    track_rows: list[dict[str, Any]] = []
    for track_id, _chosen in jobs:
        meta = plan.tracks[track_id]
        keyframes = sorted(per_crop.get(track_id, []), key=lambda row: row["t"])[:1]
        decided = _consensus_from_crops(keyframes, meta)
        track_rows.append(
            {
                "track_id": track_id,
                "label": meta.get("label"),
                "t_start": meta.get("t_start"),
                "t_end": meta.get("t_end"),
                "duration_s": meta.get("duration_s"),
                **decided,
                "keyframes": keyframes,
            }
        )
    return track_rows


def _load_engine(engine: str = "rapidocr", *, model: str | None = None):
    global _ENGINE
    if engine == "vlm":
        model_id = model or DEFAULT_VLM_MODEL
        if _ENGINE is not None and _ENGINE[0] == "vlm" and _ENGINE[1].get("model_id") == model_id:
            return _ENGINE[1]
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError(
                "VLM OCR needs extra deps. From BillboardDetect run: pip install -e '.[ocr-vlm]'"
            ) from exc
        print(f"Loading VLM {model_id} (first run downloads weights)…")
        mlx_model, processor = load(model_id)
        config = load_config(model_id)
        bundle = {"model": mlx_model, "processor": processor, "config": config, "model_id": model_id}
        _ENGINE = ("vlm", bundle)
        return bundle
    if _ENGINE is not None and _ENGINE[0] == "rapidocr":
        return _ENGINE[1]
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR needs extra deps. From BillboardDetect run: pip install -e '.[ocr]'"
        ) from exc
    rapid = RapidOCR()
    _ENGINE = ("rapidocr", rapid)
    return rapid


def _read_crop_vlm(vlm: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    import os
    import tempfile

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    if image.ndim != 3:
        raise RuntimeError("VLM crop must be a single HxWxC image.")
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    formatted = apply_chat_template(
        vlm["processor"],
        vlm["config"],
        [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": "This image is a crop of one outdoor object. Classify it with the JSON schema."},
        ],
        num_images=1,
    )
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil.save(path)
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        output = generate(
            vlm["model"],
            vlm["processor"],
            formatted,
            image=path,
            max_tokens=180,
            temperature=0.0,
            verbose=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    raw = _vlm_text(output)
    parsed = _parse_vlm_json(raw)
    parsed["raw"] = raw
    return parsed


def _consensus_from_crops(keyframes: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Label a track from exactly one crop. Overlay while consecutive boxes stay on that object."""
    empty = {
        "is_ad": False,
        "headline": "",
        "full_copy": "",
        "brand": "",
        "text": "",
        "text_conf": 0.0,
        "n_reads": len(keyframes),
        "n_agree": 0,
        "keep_t_start": meta.get("t_start"),
        "keep_t_end": meta.get("t_end"),
    }
    if not keyframes:
        return empty
    row = keyframes[0]
    if not row.get("is_ad"):
        return empty
    headline = _flatten_copy(row.get("headline") or "", max_len=80)
    full_copy = _flatten_copy(row.get("full_copy") or headline)
    keep_start, keep_end = _keep_span_matching_crop(meta, row)
    return {
        "is_ad": True,
        "headline": headline,
        "full_copy": full_copy,
        "brand": headline,
        "text": full_copy,
        "text_conf": 1.0 if full_copy else 0.0,
        "n_reads": 1,
        "n_agree": 1,
        "keep_t_start": keep_start,
        "keep_t_end": keep_end,
    }


def _keep_span_matching_crop(track: dict[str, Any], crop: dict[str, Any]) -> tuple[float, float]:
    crop_t = float(crop.get("t") or 0.0)
    boxes = sorted(track.get("boxes") or [], key=lambda box: int(box["frame"]))
    kept = _identity_chain(boxes, crop)
    if not kept:
        return round(crop_t, 3), round(crop_t, 3)
    return round(float(_box_t(kept[0])), 3), round(float(_box_t(kept[-1])), 3)


def _partition_boxes_to_crop(boxes: list[Any], crop: dict[str, Any], *, fps: float) -> tuple[list[Any], list[Any]]:
    del fps
    ordered = sorted(boxes, key=lambda box: (int(_box_frame(box)), float(_box_t(box))))
    kept = _identity_chain(ordered, crop)
    if not kept:
        if not ordered:
            return [], []
        crop_t = float(crop.get("t") or 0.0)
        nearest = min(ordered, key=lambda box: abs(float(_box_t(box)) - crop_t))
        kept = [nearest]
    kept_ids = {id(box) for box in kept}
    rest = [box for box in ordered if id(box) not in kept_ids]
    return kept, rest


def _identity_chain(boxes: list[Any], crop: dict[str, Any]) -> list[Any]:
    """Keep consecutive boxes of the same object, starting from the classified crop.

    Compare neighbors, not the frozen crop rectangle. A dashcam billboard slides and
    grows as you approach; IoU vs the first crop would cut the track while it is still
    the same ad. Stop only on a teleport between consecutive detections.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: (int(_box_frame(box)), float(_box_t(box))))
    crop_t = float(crop.get("t") or 0.0)
    real = [box for box in ordered if not _box_interpolated(box)]
    pool = real or ordered
    idx = min(range(len(pool)), key=lambda i: abs(float(_box_t(pool[i])) - crop_t))
    lo = hi = idx
    while lo > 0 and _consecutive_same_object(pool[lo], pool[lo - 1]):
        lo -= 1
    while hi + 1 < len(pool) and _consecutive_same_object(pool[hi], pool[hi + 1]):
        hi += 1
    t0 = float(_box_t(pool[lo]))
    t1 = float(_box_t(pool[hi]))
    return [box for box in ordered if t0 - 1e-6 <= float(_box_t(box)) <= t1 + 1e-6]


def _consecutive_same_object(current: Any, neighbor: Any) -> bool:
    dt = abs(float(_box_t(current)) - float(_box_t(neighbor)))
    return _boxes_continue(
        _as_xyxy(_box_xyxy(current)),
        _as_xyxy(_box_xyxy(neighbor)),
        dt=dt,
    )


def _boxes_continue(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    dt: float,
) -> bool:
    iou = _box_iou(a, b)
    if iou >= 0.15:
        return True
    ratio = max(_area(a), _area(b), 1.0) / max(min(_area(a), _area(b)), 1.0)
    if ratio > 5.0:
        return False
    dist = _center_dist(a, b)
    scale = max(_box_diag(a), _box_diag(b), 1.0)
    return dist <= 0.5 * scale + 350.0 * max(dt, 0.0)


def _contiguous_box_segments(boxes: list[Any], *, gap_s: float) -> list[list[Any]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: float(_box_t(box)))
    segments: list[list[Any]] = []
    current = [ordered[0]]
    for box in ordered[1:]:
        if float(_box_t(box)) - float(_box_t(current[-1])) > gap_s:
            segments.append(current)
            current = [box]
        else:
            current.append(box)
    segments.append(current)
    return segments


def _box_t(box: Any) -> float:
    return float(box.t if hasattr(box, "t") else box["t"])


def _box_frame(box: Any) -> int:
    return int(box.frame if hasattr(box, "frame") else box["frame"])


def _box_xyxy(box: Any) -> list[float] | tuple[float, float, float, float]:
    return box.xyxy if hasattr(box, "xyxy") else box["xyxy"]


def _box_interpolated(box: Any) -> bool:
    if hasattr(box, "interpolated"):
        return bool(box.interpolated)
    return bool(box.get("interpolated")) if isinstance(box, dict) else False


def _center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _box_diag(box: tuple[float, float, float, float]) -> float:
    return max(1.0, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _vlm_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    text = getattr(output, "text", None)
    if text:
        return str(text)
    return str(output)


def _parse_vlm_json(raw: str) -> dict[str, Any]:
    blob = raw.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    data = _load_json_object(blob)
    if data is None:
        full_copy = (
            _extract_json_string_field(blob, "full_copy")
            or _extract_json_string_field(blob, "text")
            or ""
        )
        headline = (
            _extract_json_string_field(blob, "headline")
            or _extract_json_string_field(blob, "brand")
            or ""
        )
        is_raw = _extract_json_bool_field(blob, "is_billboard")
        if is_raw is None:
            is_raw = _extract_json_bool_field(blob, "is_ad")
        return {
            "is_ad": is_raw if is_raw is not None else not _is_not_ad_text(full_copy),
            "headline": headline,
            "full_copy": full_copy,
            "brand": headline,
            "text": full_copy,
        }
    headline = str(_first_value(data, "headline", "brand", "brand/headline") or "")
    full_copy = str(_first_value(data, "full_copy", "text", "copy") or "")
    is_ad = _first_value(data, "is_billboard", "is_ad", "is ad/billboard", "is_ad/billboard")
    if is_ad is None:
        is_ad = bool(full_copy) and not _is_not_ad_text(full_copy)
    return {
        "is_ad": _truthy(is_ad),
        "headline": headline,
        "full_copy": full_copy,
        "brand": headline,
        "text": full_copy,
    }


def _first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    lower = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key.lower() in lower and lower[key.lower()] is not None:
            return lower[key.lower()]
    return None


def _extract_json_bool_field(blob: str, key: str) -> bool | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false|"true"|"false"|1|0)', blob, re.I)
    if not match:
        return None
    return _truthy(match.group(1).strip('"'))


def _load_json_object(blob: str) -> dict[str, Any] | None:
    start = blob.find("{")
    if start < 0:
        return None
    try:
        data, _end = json.JSONDecoder().raw_decode(blob[start:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_json_string_field(blob: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', blob)
    if not match:
        return None
    chars: list[str] = []
    i = match.end()
    escapes = {"n": "\n", "t": " ", "r": "", '"': '"', "\\": "\\"}
    while i < len(blob):
        ch = blob[i]
        if ch == "\\" and i + 1 < len(blob):
            chars.append(escapes.get(blob[i + 1], blob[i + 1]))
            i += 2
            continue
        if ch == '"':
            break
        chars.append(ch)
        i += 1
    return "".join(chars)


def _flatten_copy(value: object, *, max_len: int = 400) -> str:
    text = str(value or "")
    stripped = text.strip()
    if stripped.startswith("{") and '"text"' in stripped:
        extracted = _extract_json_string_field(stripped, "text")
        if extracted:
            text = extracted
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip(" \t\"'")
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip(",;:")
    return text


def _read_crop(engine, image: np.ndarray) -> list[dict[str, Any]]:
    result = engine(image)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    boxes = getattr(result, "boxes", None)
    if not txts:
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for i, text in enumerate(txts):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        conf = float(scores[i]) if scores is not None and i < len(scores) and scores[i] is not None else 0.0
        y = 0.0
        if boxes is not None and i < len(boxes) and boxes[i] is not None:
            pts = np.asarray(boxes[i], dtype=np.float32).reshape(-1, 2)
            y = float(pts[:, 1].min()) if pts.size else 0.0
        rows.append((y, {"text": cleaned, "conf": round(conf, 4)}))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]


def _vote(reads: list[tuple[str, float, float]]) -> tuple[str, float, int]:
    if not reads:
        return "", 0.0, 0
    clusters: list[list[tuple[str, float, float]]] = []
    for text, conf, area in reads:
        placed = False
        for cluster in clusters:
            if _similar(_normalize(text), _normalize(cluster[0][0])) >= VOTE_SIMILARITY:
                cluster.append((text, conf, area))
                placed = True
                break
        if not placed:
            clusters.append([(text, conf, area)])
    clusters.sort(key=lambda cluster: (len(cluster), sum(item[1] for item in cluster)), reverse=True)
    best = clusters[0]
    n_agree = len(best)
    if n_agree == 1 and (len(reads) > 1 and best[0][1] < SINGLE_READ_MIN_CONF):
        largest = max(reads, key=lambda item: item[2])
        if largest[1] >= SINGLE_READ_MIN_CONF:
            return largest[0], largest[1], 1
        return "", 0.0, 0
    text = max(best, key=lambda item: (item[1], item[2]))[0]
    conf = sum(item[1] for item in best) / n_agree
    return text, conf, n_agree


def _spread(boxes: list[Any], *, fps: float, min_spacing_s: float, limit: int) -> list[Any]:
    min_frames = max(1, int(round(min_spacing_s * fps)))
    picked: list[Any] = []
    for box in boxes:
        frame = int(box["frame"]) if isinstance(box, dict) else int(box.frame)
        if any(abs(frame - (int(p["frame"]) if isinstance(p, dict) else int(p.frame))) < min_frames for p in picked):
            continue
        picked.append(box)
        if len(picked) >= limit:
            break
    return sorted(picked, key=lambda item: int(item["frame"]) if isinstance(item, dict) else int(item.frame))


def _pad_xyxy(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width), x2 + dx),
        min(float(height), y2 + dy),
    )


def _upscale(crop: np.ndarray, short_side: int) -> np.ndarray:
    h, w = crop.shape[:2]
    shortest = min(h, w)
    if shortest >= short_side:
        return crop
    scale = short_side / shortest
    return cv2.resize(crop, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)


def _sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _as_xyxy(raw: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _area(xyxy: list[float] | tuple[float, float, float, float]) -> float:
    return max(0.0, float(xyxy[2]) - float(xyxy[0])) * max(0.0, float(xyxy[3]) - float(xyxy[1]))


def _min_side(xyxy: list[float]) -> float:
    if len(xyxy) < 4:
        return 0.0
    return min(max(0.0, float(xyxy[2]) - float(xyxy[0])), max(0.0, float(xyxy[3]) - float(xyxy[1])))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_not_ad_text(text: str) -> bool:
    return _normalize(text) in _NOT_AD_MARKERS


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]

