from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    total_frames: int | None


def probe(path: Path) -> VideoMeta:
    ffprobe = _tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(_ffmpeg_hint(path, result.stderr))
    streams = json.loads(result.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(_ffmpeg_hint(path, "no video stream"))
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 24.0
    total = _parse_int(stream.get("nb_frames"))
    if total is None:
        duration = _parse_float(stream.get("duration"))
        if duration is not None and fps > 0:
            total = max(1, int(round(duration * fps)))
    if width <= 0 or height <= 0:
        raise RuntimeError(_ffmpeg_hint(path, "invalid frame size"))
    return VideoMeta(width=width, height=height, fps=fps, total_frames=total)


def iter_frames(path: Path, meta: VideoMeta) -> Iterator[np.ndarray]:
    ffmpeg = _tool("ffmpeg")
    frame_bytes = meta.width * meta.height * 3
    proc = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise RuntimeError(_ffmpeg_hint(path, "truncated frame from ffmpeg"))
            yield np.frombuffer(buf, dtype=np.uint8).reshape((meta.height, meta.width, 3)).copy()
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(_ffmpeg_hint(path, stderr.decode("utf-8", errors="replace")))


class Mp4Writer:
    """Writes H.264 MP4 via system ffmpeg (libx264, yuv420p, faststart)."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.src_width = width
        self.src_height = height
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        self.fps = fps if fps > 0 else 24.0
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Mp4Writer:
        ffmpeg = _tool("ffmpeg")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Mp4Writer is not open")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = _pad_even(frame, self.width, self.height)
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read() if self._proc.stderr else b""
        rc = self._proc.wait()
        self._proc = None
        if exc_type is None and rc != 0:
            raise RuntimeError(
                f"ffmpeg failed writing {self.path}: {stderr.decode('utf-8', errors='replace').strip() or rc}"
            )


def _pad_even(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    h = min(frame.shape[0], height)
    w = min(frame.shape[1], width)
    out[:h, :w] = frame[:h, :w]
    return out


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH. Install with: brew install ffmpeg")
    return path


def _parse_rate(value: object) -> float | None:
    if not value or value == "0/0" or value == "N/A":
        return None
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        denom = float(den)
        if denom == 0:
            return None
        return float(num) / denom
    return _parse_float(text)


def _parse_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ffmpeg_hint(path: Path, detail: str = "") -> str:
    extra = f" ({detail.strip()})" if detail and detail.strip() else ""
    return (
        f"Could not read {path}{extra}. GlassesCapture writes HEVC .mov files; this pipeline "
        "uses system ffmpeg to decode them and write H.264 MP4. Install ffmpeg (brew install ffmpeg) "
        "and retry."
    )
