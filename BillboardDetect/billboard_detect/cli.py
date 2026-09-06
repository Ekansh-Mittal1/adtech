from __future__ import annotations

import argparse
import sys
from pathlib import Path

from billboard_detect.detectors import DEFAULT_CLASSES, build_detector, pick_device
from billboard_detect.ocr import DEFAULT_VLM_MODEL
from billboard_detect.pipeline import clip_output_base, collect_videos, process_from_tracks, process_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and track billboards/advertisements in a video or folder of clips.",
    )
    parser.add_argument("source", type=Path, help="Video file or folder of clips (.mov, .mp4, …)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Parent directory for per-clip results (default: ./output). Each run writes to <parent>/<stem>/v001, v002, … and never overwrites a previous run.",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo-world", "grounding-dino"),
        default="yolo-world",
        help="Open-vocabulary detector (default: yolo-world)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Text queries to detect (default: billboard / digital / outdoor / LED billboard, advertisement, poster, banner, truck-side ad, vehicle wrap)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        nargs="+",
        default=[0.25],
        help="One or more confidence thresholds (default: 0.25). The model runs once at the lowest value.",
    )
    parser.add_argument(
        "--from-detections",
        type=Path,
        default=None,
        help="Replay a saved *_detections.json and skip the model. Only thresholds >= the saved detect_conf work.",
    )
    parser.add_argument(
        "--from-tracks",
        type=Path,
        default=None,
        help="Reuse a saved *_tracks.json (skip detect and track). Re-annotates; add --ocr to read billboard text.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="OCR lasting tracks from real (non-interpolated) keyframes.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=("rapidocr", "vlm"),
        default="rapidocr",
        help="rapidocr is fast local OCR; vlm is Qwen2.5-VL on Apple Silicon (better on logos/stylized type).",
    )
    parser.add_argument(
        "--ocr-vlm-model",
        default=DEFAULT_VLM_MODEL,
        help=f"HuggingFace MLX model id when --ocr-engine vlm (default: {DEFAULT_VLM_MODEL})",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="With --from-tracks, skip rewriting the annotated MP4 (still decodes frames for OCR crops).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Run detection every Nth frame (default: 2). Annotated video still includes every frame.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Interpolate a track across missed frames shorter than this (default: 1.0). 0 disables.",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="EMA-smooth box corners across time (default: on). Use --no-smooth to disable.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.4,
        metavar="ALPHA",
        help="EMA blending when --smooth is on; lower is smoother (default: 0.4).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO-World weights file, or HuggingFace model id for Grounding DINO",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO-World inference size (default: 1280). Use 1920 for extra 4K recall.",
    )
    parser.add_argument("--device", default=None, help="torch device: mps, cuda, or cpu (default: auto)")
    args = parser.parse_args(argv)

    try:
        if args.max_gap < 0:
            raise RuntimeError(f"--max-gap must be >= 0, got {args.max_gap}")
        if args.smooth_alpha <= 0 or args.smooth_alpha > 1:
            raise RuntimeError(f"--smooth-alpha must be in (0, 1], got {args.smooth_alpha}")
        if args.from_tracks is not None and args.from_detections is not None:
            raise RuntimeError("Use only one of --from-tracks or --from-detections.")
        if args.ocr_engine == "vlm":
            args.ocr = True
        if args.no_annotate and args.from_tracks is None:
            raise RuntimeError("--no-annotate only applies with --from-tracks.")
        videos = collect_videos(args.source)
        if args.from_tracks is not None:
            if len(videos) != 1:
                raise RuntimeError("--from-tracks requires a single source video, not a folder.")
            if not args.ocr and args.no_annotate:
                raise RuntimeError("--from-tracks --no-annotate needs --ocr (or --ocr-engine vlm).")
            dest = clip_output_base(args.output, videos[0].stem)
            print(f"Reusing tracks from {args.from_tracks}" + (" + OCR" if args.ocr else ""))
            process_from_tracks(
                videos[0],
                args.from_tracks,
                dest,
                ocr=args.ocr,
                ocr_engine=args.ocr_engine,
                ocr_vlm_model=args.ocr_vlm_model,
                annotate=not args.no_annotate,
            )
            return 0
        if args.from_detections is not None:
            if len(videos) != 1:
                raise RuntimeError("--from-detections requires a single source video, not a folder.")
            detector = None
            print(f"Skipping model; exporting {len(args.conf)} confidence level(s) from {args.from_detections}")
        else:
            detect_conf = min(args.conf)
            detector = build_detector(
                args.detector,
                classes=args.classes,
                conf=detect_conf,
                device=args.device,
                weights=args.weights,
                imgsz=args.imgsz,
            )
            device = pick_device(args.device)
            print(
                f"Detector {detector.name} on {device}; {len(videos)} clip(s); "
                f"detect_conf={detect_conf:g}; export confs={', '.join(f'{c:g}' for c in args.conf)}"
            )
        for video in videos:
            process_video(
                video,
                clip_output_base(args.output, video.stem),
                detector=detector,
                confs=args.conf,
                frame_stride=args.frame_stride,
                from_detections=args.from_detections,
                max_gap_s=args.max_gap,
                smooth_alpha=args.smooth_alpha if args.smooth else 1.0,
                ocr=args.ocr,
                ocr_engine=args.ocr_engine,
                ocr_vlm_model=args.ocr_vlm_model,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0
