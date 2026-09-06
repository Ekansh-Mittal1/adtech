from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

from billboard_detect.detectors import Detector, Detection
from billboard_detect.export import (
    BoxObs,
    TrackBook,
    filter_detections,
    load_raw_detections,
    load_tracks_json,
    write_appearances_csv,
    nms_detections,
    write_raw_detections,
    write_tracks_json,
)
from billboard_detect.smooth import split_jumped_tracks, stabilize_tracks, stitch_broken_tracks
from billboard_detect.video_io import Mp4Writer, VideoMeta, iter_frames, probe

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
_RUN_DIR_RE = None


def allocate_run_dir(base: Path) -> Path:
    """Create the next unused ``<base>/v001``, ``v002``, … so a run never overwrites another."""
    import re

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    highest = 0
    for child in base.iterdir():
        match = re.fullmatch(r"v(\d+)", child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    n = highest + 1
    while n <= 9999:
        dest = base / f"v{n:03d}"
        try:
            dest.mkdir(exist_ok=False)
            return dest
        except FileExistsError:
            n += 1
    raise RuntimeError(f"No free run folder under {base} (v001–v9999 are taken).")


def clip_output_base(output: Path | None, stem: str) -> Path:
    """Folder that holds v001, v002, … for one clip."""
    root = Path(output) if output is not None else Path("output")
    if root.name == stem:
        return root
    return root / stem


def collect_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"No file or folder at {source}")
    videos = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files in {source}")
    return videos


def process_video(
    source: Path,
    output_dir: Path,
    *,
    detector: Detector | None,
    confs: list[float],
    frame_stride: int,
    from_detections: Path | None = None,
    max_gap_s: float = 1.0,
    smooth_alpha: float = 0.4,
    ocr: bool = False,
    ocr_engine: str = "rapidocr",
    ocr_vlm_model: str | None = None,
) -> Path:
    source = source.resolve()
    output_dir = allocate_run_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_dir}")
    confs = _unique_confs(confs)
    detect_conf = min(confs)
    stride = max(1, frame_stride)
    if ocr:
        from billboard_detect.ocr import ensure_engine

        ensure_engine(ocr_engine, model=ocr_vlm_model)

    if from_detections is not None:
        raw, by_frame = load_raw_detections(from_detections)
        meta = probe(source)
        fps = float(raw.get("fps") or meta.fps)
        width = int(raw.get("width") or meta.width)
        height = int(raw.get("height") or meta.height)
        n_frames = int(raw.get("n_frames") or meta.total_frames or 0)
        detector_name = str(raw.get("detector") or (detector.name if detector else "unknown"))
        classes = list(raw.get("classes") or (detector.classes if detector else []))
        stored_conf = float(raw.get("detect_conf") or detect_conf)
        stride = int(raw.get("frame_stride") or stride)
        if detect_conf + 1e-9 < stored_conf:
            print(
                f"Warning: detections were saved at conf {stored_conf:g}; "
                f"cannot recover boxes below that. Requested min {detect_conf:g}."
            )
        print(f"Replaying detections from {from_detections} ({len(by_frame)} keyed frames)")
    else:
        if detector is None:
            raise RuntimeError("A detector is required unless --from-detections or --from-tracks is set.")
        meta = probe(source)
        fps = meta.fps
        width, height = meta.width, meta.height
        detector_name = detector.name
        classes = list(detector.classes)
        by_frame, n_frames = _run_detector(source, meta, detector, stride)
        raw_path = output_dir / f"{source.stem}_detections.json"
        write_raw_detections(
            raw_path,
            source=str(source),
            fps=fps,
            width=width,
            height=height,
            detector=detector_name,
            classes=classes,
            detect_conf=detect_conf,
            frame_stride=stride,
            n_frames=n_frames,
            by_frame=by_frame,
        )
        print(f"Wrote {raw_path} (detect_conf={detect_conf:g}; replay with --from-detections)")

    last_path = output_dir
    for conf in confs:
        dest = output_dir if len(confs) == 1 else output_dir / f"conf_{_conf_tag(conf)}"
        dest.mkdir(parents=True, exist_ok=True)
        last_path = _export_threshold(
            source,
            dest,
            meta=VideoMeta(width=width, height=height, fps=fps, total_frames=n_frames or meta.total_frames),
            by_frame=filter_detections(by_frame, conf),
            classes=classes,
            detector_name=detector_name,
            conf=conf,
            frame_stride=stride,
            n_frames=n_frames,
            max_gap_s=max_gap_s,
            smooth_alpha=smooth_alpha,
            ocr=ocr,
            ocr_engine=ocr_engine,
            ocr_vlm_model=ocr_vlm_model,
        )
    return last_path


def process_from_tracks(
    source: Path,
    tracks_path: Path,
    output_dir: Path,
    *,
    ocr: bool = False,
    ocr_engine: str = "rapidocr",
    ocr_vlm_model: str | None = None,
    annotate: bool = True,
) -> Path:
    """Re-annotate (and optionally OCR) from a saved *_tracks.json. Does not re-run the detector."""
    source = source.resolve()
    tracks_path = tracks_path.resolve()
    output_dir = allocate_run_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_dir}")
    if ocr:
        from billboard_detect.ocr import ensure_engine

        ensure_engine(ocr_engine, model=ocr_vlm_model)
    payload, book = load_tracks_json(tracks_path)
    meta = probe(source)
    fps = float(payload.get("fps") or meta.fps or 24.0)
    width = int(payload.get("width") or meta.width)
    height = int(payload.get("height") or meta.height)
    if (width, height) != (meta.width, meta.height):
        print(
            f"Warning: tracks.json is {width}x{height} but {source.name} is "
            f"{meta.width}x{meta.height}. Crops may be misaligned."
        )
        width, height = meta.width, meta.height
    classes = list(payload.get("classes") or [])
    video_meta = VideoMeta(width=width, height=height, fps=fps, total_frames=meta.total_frames)
    annotated_path = output_dir / f"{source.stem}_annotated.mp4"
    _write_annotated(
        source,
        annotated_path,
        book=book,
        meta=video_meta,
        classes=classes,
        ocr=ocr,
        ocr_engine=ocr_engine,
        ocr_vlm_model=ocr_vlm_model,
        annotate=annotate,
        tracks=book.summaries(),
        tracks_file=str(tracks_path),
        desc=f"{source.name} tracks",
    )
    if ocr:
        json_path = output_dir / f"{source.stem}_tracks.json"
        csv_path = output_dir / f"{source.stem}_appearances.csv"
        tracks = book.summaries()
        write_tracks_json(json_path, {**payload, "tracks": tracks, "vlm_cuts": True})
        write_appearances_csv(csv_path, tracks)
        print(f"Wrote {json_path} ({len(tracks)} tracks after VLM cuts)")
        print(f"Wrote {csv_path}")
    if annotate:
        print(f"Wrote {annotated_path} from {tracks_path}")
    return annotated_path


def _run_detector(
    source: Path,
    meta: VideoMeta,
    detector: Detector,
    stride: int,
) -> tuple[dict[int, list[Detection]], int]:
    by_frame: dict[int, list[Detection]] = {}
    n_frames = 0
    for index, frame in enumerate(
        tqdm(iter_frames(source, meta), total=meta.total_frames, desc=f"{source.name} detect", unit="f")
    ):
        n_frames += 1
        if index % stride == 0:
            by_frame[index] = detector.detect(frame)
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {source}")
    return by_frame, n_frames


def _export_threshold(
    source: Path,
    output_dir: Path,
    *,
    meta: VideoMeta,
    by_frame: dict[int, list[Detection]],
    classes: list[str],
    detector_name: str,
    conf: float,
    frame_stride: int,
    n_frames: int,
    max_gap_s: float,
    smooth_alpha: float,
    ocr: bool = False,
    ocr_engine: str = "rapidocr",
    ocr_vlm_model: str | None = None,
) -> Path:
    fps = meta.fps if meta.fps else 24.0
    total = n_frames or meta.total_frames or 0
    if total <= 0:
        keyed = max(by_frame, default=-1) + 1
        total = keyed
    tracker_fps = max(1.0, fps / max(1, frame_stride))
    tracker = sv.ByteTrack(
        frame_rate=tracker_fps,
        track_activation_threshold=max(conf, 0.05),
        lost_track_buffer=max(8, int(round(0.35 * tracker_fps))),
        minimum_matching_threshold=0.6,
        minimum_consecutive_frames=1,
    )
    book = _track_detections(
        by_frame,
        tracker,
        classes,
        fps=fps,
        frame_stride=frame_stride,
        n_frames=total,
        frame_area=float(meta.width * meta.height),
    )
    split_jumped_tracks(book, fps=fps)
    stitch_broken_tracks(book, fps=fps, max_gap_s=max_gap_s)
    stabilize_tracks(
        book,
        fps=fps,
        max_gap_s=max_gap_s,
        frame_stride=frame_stride,
        smooth_alpha=smooth_alpha,
        width=meta.width,
        height=meta.height,
    )

    annotated_path = output_dir / f"{source.stem}_annotated.mp4"
    json_path = output_dir / f"{source.stem}_tracks.json"
    csv_path = output_dir / f"{source.stem}_appearances.csv"
    tracks = book.summaries()
    wrote = _write_annotated(
        source,
        annotated_path,
        book=book,
        meta=meta,
        classes=classes,
        ocr=ocr,
        ocr_engine=ocr_engine,
        ocr_vlm_model=ocr_vlm_model,
        tracks=tracks,
        tracks_file=str(json_path),
        desc=f"{source.name} conf={conf:g}",
    )

    if wrote == 0:
        raise RuntimeError(f"No frames decoded from {source}")

    tracks = book.summaries()

    payload = {
        "source": str(source.resolve()),
        "fps": fps,
        "width": meta.width,
        "height": meta.height,
        "detector": detector_name,
        "classes": classes,
        "confidence_threshold": conf,
        "frame_stride": frame_stride,
        "n_frames": n_frames or wrote,
        "max_gap_s": max_gap_s,
        "smooth_alpha": smooth_alpha,
        "tracks": tracks,
    }
    write_tracks_json(json_path, payload)
    write_appearances_csv(csv_path, tracks)
    print(f"Wrote {annotated_path}")
    print(f"Wrote {json_path} ({len(tracks)} tracks, conf={conf:g})")
    print(f"Wrote {csv_path}")
    return annotated_path


def _write_annotated(
    source: Path,
    annotated_path: Path,
    *,
    book: TrackBook,
    meta: VideoMeta,
    classes: list[str],
    ocr: bool,
    tracks: list[dict],
    tracks_file: str,
    desc: str,
    ocr_engine: str = "rapidocr",
    ocr_vlm_model: str | None = None,
    annotate: bool = True,
) -> int:
    fps = meta.fps if meta.fps else 24.0
    overlay_texts: dict[int, str] | None = None
    overlay_spans: dict[int, tuple[float, float]] | None = None
    ocr_payload: dict | None = None
    ocr_path = annotated_path.parent / f"{source.stem}_ocr.json"
    wrote = 0

    if ocr:
        from billboard_detect.ocr import (
            apply_vlm_track_cuts,
            crop_short_side,
            grab_crop,
            overlay_labels,
            overlay_windows,
            plan_ocr,
            run_ocr,
        )

        short_side = crop_short_side(ocr_engine)
        plan = plan_ocr(tracks, width=meta.width, height=meta.height, fps=fps)
        print(
            f"OCR ({ocr_engine}): {len(plan.tracks)} tracks, {len(plan.requests)} candidate crops "
            f"(real detections only, duration >= 0.05s)"
        )
        crops: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        for index, frame in enumerate(
            tqdm(iter_frames(source, meta), total=meta.total_frames, desc=f"{desc} crops", unit="f")
        ):
            for req in plan.by_frame.get(index, []):
                grabbed = grab_crop(frame, req.padded_xyxy, short_side=short_side)
                if grabbed is not None:
                    crops[(req.track_id, req.frame)] = grabbed
            wrote += 1
        ocr_payload = run_ocr(
            plan,
            crops,
            source=str(source.resolve()),
            tracks_file=tracks_file,
            fps=fps,
            width=meta.width,
            height=meta.height,
            engine=ocr_engine,
            vlm_model=ocr_vlm_model,
        )
        n_split = apply_vlm_track_cuts(book, ocr_payload, fps=fps)
        cuts = ocr_payload.get("vlm_track_cuts") or {}
        print(
            f"VLM track cuts: trimmed {cuts.get('trimmed', 0)} labeled tracks, "
            f"split {n_split} leftover fragments into new unlabeled IDs"
        )
        overlay_texts = overlay_labels(ocr_payload)
        overlay_spans = overlay_windows(ocr_payload)
        _write_ocr_files(annotated_path.parent, source.stem, ocr_payload)
    elif ocr_path.is_file():
        from billboard_detect.ocr import apply_vlm_track_cuts, load_ocr_json, overlay_labels, overlay_windows

        ocr_payload = load_ocr_json(ocr_path)
        apply_vlm_track_cuts(book, ocr_payload, fps=fps)
        overlay_texts = overlay_labels(ocr_payload)
        overlay_spans = overlay_windows(ocr_payload)
        _write_ocr_files(annotated_path.parent, source.stem, ocr_payload)
        print(f"Overlay labels from {ocr_path} ({len(overlay_texts)} tracks with text)")

    indexed = book.by_frame()

    if annotate:
        ads_path = None
        ads_ids = None
        if ocr_payload is not None:
            from billboard_detect.ocr import ad_track_ids

            ads_ids = ad_track_ids(ocr_payload)
            ads_path = annotated_path.with_name(f"{source.stem}_annotated_ads.mp4")
        wrote = _render_annotated(
            source,
            annotated_path,
            indexed=indexed,
            meta=meta,
            classes=classes,
            desc=desc,
            overlay_texts=overlay_texts,
            overlay_spans=overlay_spans,
            ads_path=ads_path,
            ads_ids=ads_ids,
        )
        if ads_path is not None:
            print(f"Wrote {ads_path} ({len(ads_ids or ())} VLM/OCR ad tracks)")
    return wrote


def _write_ocr_files(output_dir: Path, stem: str, payload: dict) -> None:
    from billboard_detect.ocr import write_ocr_outputs

    json_path, csv_path = write_ocr_outputs(output_dir, stem, payload)
    n_tracks = int(payload.get("n_tracks") or len(payload.get("tracks") or []))
    n_text = sum(1 for track in payload.get("tracks") or [] if track.get("full_copy") or track.get("text"))
    n_ads = sum(1 for track in payload.get("tracks") or [] if track.get("is_ad"))
    print(f"Wrote {json_path} ({n_text}/{n_tracks} tracks with text, {n_ads} ads)")
    print(f"Wrote {csv_path}")


def _render_annotated(
    source: Path,
    annotated_path: Path,
    *,
    indexed: dict[int, list],
    meta: VideoMeta,
    classes: list[str],
    desc: str,
    overlay_texts: dict[int, str] | None,
    overlay_spans: dict[int, tuple[float, float]] | None = None,
    ads_path: Path | None = None,
    ads_ids: set[int] | None = None,
) -> int:
    fps = meta.fps if meta.fps else 24.0
    outline, box_annotator, label_annotator = _make_annotators(meta.width, meta.height)
    wrote = 0
    with ExitStack() as stack:
        all_sink = stack.enter_context(Mp4Writer(annotated_path, meta.width, meta.height, fps))
        ads_sink = (
            stack.enter_context(Mp4Writer(ads_path, meta.width, meta.height, fps))
            if ads_path is not None
            else None
        )
        for index, frame in enumerate(
            tqdm(iter_frames(source, meta), total=meta.total_frames, desc=desc, unit="f")
        ):
            t = index / fps
            items = indexed.get(index, [])
            all_sink.write(
                _annotate(
                    frame,
                    _obs_to_sv(items, classes),
                    classes,
                    outline,
                    box_annotator,
                    label_annotator,
                    overlay_texts=overlay_texts,
                    overlay_spans=overlay_spans,
                    t=t,
                )
            )
            if ads_sink is not None:
                ads_sink.write(
                    _annotate(
                        frame,
                        _obs_to_sv(items, classes, keep_ids=ads_ids, t=t, windows=overlay_spans),
                        classes,
                        outline,
                        box_annotator,
                        label_annotator,
                        overlay_texts=overlay_texts,
                        overlay_spans=overlay_spans,
                        t=t,
                    )
                )
            wrote += 1
    return wrote


def _make_annotators(
    width: int, height: int
) -> tuple[sv.BoxAnnotator, sv.BoxAnnotator, sv.LabelAnnotator]:
    short = max(1, min(width, height))
    orig_thickness = 2
    orig_text_scale = 0.5
    orig_text_thickness = 1
    orig_padding = 10
    thickness = max(6, int(round(short * 0.0055)))
    outline_thickness = max(thickness + 5, int(round(short * 0.008)))
    text_scale = max(0.7, short / 1080.0 * 0.75)
    text_thickness = max(2, thickness // 3)
    text_padding = max(12, thickness * 2)
    thickness = max(orig_thickness + 2, int(round(thickness * 0.8)))
    outline_thickness = max(thickness + 3, int(round(outline_thickness * 0.8)))
    text_scale = (orig_text_scale + text_scale) / 2
    text_thickness = max(1, int(round((orig_text_thickness + text_thickness) / 2)))
    text_padding = max(orig_padding, int(round((orig_padding + text_padding) / 2)))
    color = sv.Color.from_hex("#FFE600")
    ink = sv.Color.from_hex("#111111")
    outline = sv.BoxAnnotator(color=ink, thickness=outline_thickness)
    boxes = sv.BoxAnnotator(color=color, thickness=thickness)
    labels = sv.LabelAnnotator(
        color=color,
        text_color=ink,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
    )
    return outline, boxes, labels


def _track_detections(
    by_frame: dict[int, list[Detection]],
    tracker: sv.ByteTrack,
    classes: list[str],
    *,
    fps: float,
    frame_stride: int,
    n_frames: int,
    frame_area: float | None = None,
) -> TrackBook:
    book = TrackBook()
    stride = max(1, frame_stride)
    last = max(n_frames, max(by_frame, default=-1) + 1)
    for index in range(0, last, stride):
        detections = _to_sv(
            nms_detections(by_frame.get(index, []), frame_area=frame_area),
            classes,
        )
        detections = tracker.update_with_detections(detections)
        _record(book, detections, classes, frame=index, t=index / fps)
    return book


def _unique_confs(confs: list[float]) -> list[float]:
    ordered: list[float] = []
    seen: set[float] = set()
    for conf in confs:
        if conf < 0 or conf > 1:
            raise RuntimeError(f"Confidence must be between 0 and 1, got {conf}")
        key = round(conf, 6)
        if key not in seen:
            seen.add(key)
            ordered.append(conf)
    if not ordered:
        raise RuntimeError("Pass at least one --conf value.")
    return ordered


def _conf_tag(conf: float) -> str:
    return f"{conf:g}".replace(".", "p")


def _to_sv(detections: list[Detection], classes: list[str]) -> sv.Detections:
    if not detections:
        return _empty_detections()
    xyxy = np.array([det.xyxy for det in detections], dtype=np.float32)
    confidence = np.array([det.confidence for det in detections], dtype=np.float32)
    class_id = np.array([_class_id(det.label, classes) for det in detections], dtype=int)
    names = np.array([det.label for det in detections], dtype=object)
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": names},
    )


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.array([], dtype=np.float32),
        class_id=np.array([], dtype=int),
    )


def _class_id(label: str, classes: list[str]) -> int:
    try:
        return classes.index(label)
    except ValueError:
        return 0


def _obs_to_sv(
    items: list[tuple[int, BoxObs]],
    classes: list[str],
    keep_ids: set[int] | None = None,
    t: float | None = None,
    windows: dict[int, tuple[float, float]] | None = None,
) -> sv.Detections:
    if keep_ids is not None:
        items = [(track_id, box) for track_id, box in items if track_id in keep_ids]
    if windows is not None and t is not None:
        items = [(track_id, box) for track_id, box in items if _in_overlay_window(track_id, t, windows)]
    if not items:
        return _empty_detections()
    xyxy = np.array([box.xyxy for _, box in items], dtype=np.float32)
    confidence = np.array([box.conf for _, box in items], dtype=np.float32)
    class_id = np.array([_class_id(box.label, classes) for _, box in items], dtype=int)
    names = np.array([box.label for _, box in items], dtype=object)
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": names},
    )
    detections.tracker_id = np.array([track_id for track_id, _ in items], dtype=int)
    return detections


def _record(book: TrackBook, detections: sv.Detections, classes: list[str], *, frame: int, t: float) -> None:
    if detections.tracker_id is None or len(detections) == 0:
        return
    names = detections.data.get("class_name") if detections.data else None
    for i in range(len(detections)):
        track_id = detections.tracker_id[i]
        if track_id is None:
            continue
        xyxy = tuple(float(v) for v in detections.xyxy[i])
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        if names is not None:
            label = str(names[i])
        else:
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            label = classes[cls_id] if 0 <= cls_id < len(classes) else "advertisement"
        book.add(int(track_id), frame=frame, t=t, xyxy=xyxy, conf=conf, label=label)


def _in_overlay_window(track_id: int, t: float, windows: dict[int, tuple[float, float]]) -> bool:
    span = windows.get(int(track_id))
    if span is None:
        return False
    return span[0] - 1e-6 <= t <= span[1] + 1e-6


def _annotate(
    frame: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    outline: sv.BoxAnnotator,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    overlay_texts: dict[int, str] | None = None,
    overlay_spans: dict[int, tuple[float, float]] | None = None,
    t: float | None = None,
) -> np.ndarray:
    scene = frame.copy()
    if len(detections) == 0:
        return scene
    labels = []
    names = detections.data.get("class_name") if detections.data else None
    for i in range(len(detections)):
        track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
        tid = int(track_id) if track_id is not None else None
        caption = ""
        if overlay_texts is not None and tid is not None:
            if overlay_spans is None or t is None or _in_overlay_window(tid, t, overlay_spans):
                caption = overlay_texts.get(tid, "")
        shown = f"#{tid}" if tid is not None else "#?"
        if overlay_texts is not None:
            labels.append(f"{shown} {caption}".rstrip() if caption else shown)
            continue
        if names is not None:
            name = str(names[i])
        else:
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            name = classes[cls_id] if 0 <= cls_id < len(classes) else "ad"
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        labels.append(f"{shown} {name} {conf:.2f}")
    scene = outline.annotate(scene=scene, detections=detections)
    scene = box_annotator.annotate(scene=scene, detections=detections)
    return label_annotator.annotate(scene=scene, detections=detections, labels=labels)
