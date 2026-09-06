from __future__ import annotations

from billboard_detect.export import BoxObs, TrackBook


def split_jumped_tracks(book: TrackBook, *, fps: float) -> None:
    """Break an ID when consecutive hits teleport onto a different object."""
    if fps <= 0 or not book.observations:
        return
    next_id = max(book.observations) + 1
    updated: dict[int, list[BoxObs]] = {}
    for track_id, boxes in book.observations.items():
        ordered = sorted(boxes, key=lambda box: box.frame)
        if not ordered:
            continue
        segments: list[list[BoxObs]] = [[ordered[0]]]
        for box in ordered[1:]:
            prev = segments[-1][-1]
            gap = box.frame - prev.frame
            if _same_identity(prev.xyxy, box.xyxy, gap_frames=gap, fps=fps):
                segments[-1].append(box)
            else:
                segments.append([box])
        updated[int(track_id)] = segments[0]
        for segment in segments[1:]:
            updated[next_id] = segment
            next_id += 1
    book.observations.clear()
    book.observations.update(updated)


def stitch_broken_tracks(book: TrackBook, *, fps: float, max_gap_s: float = 0.4) -> None:
    """Merge sequential IDs only when the box still looks like the same board."""
    if fps <= 0:
        return
    max_gap_s = max(0.0, min(max_gap_s, 0.4))
    items: list[tuple[int, list[BoxObs]]] = []
    for track_id, boxes in book.observations.items():
        if boxes:
            items.append((int(track_id), sorted(boxes, key=lambda box: box.frame)))
    items.sort(key=lambda row: row[1][0].frame)
    absorbed: set[int] = set()
    for index, (id_a, boxes_a) in enumerate(items):
        if id_a in absorbed:
            continue
        changed = True
        while changed:
            changed = False
            end_a = boxes_a[-1]
            for id_b, boxes_b in items[index + 1 :]:
                if id_b in absorbed:
                    continue
                start_b = boxes_b[0]
                gap_frames = start_b.frame - end_a.frame
                gap_s = gap_frames / fps
                if gap_s < -0.05 or gap_s > max_gap_s:
                    continue
                if not _same_identity(end_a.xyxy, start_b.xyxy, gap_frames=max(gap_frames, 1), fps=fps):
                    continue
                occupied = {box.frame for box in boxes_a}
                boxes_a = sorted(boxes_a + [box for box in boxes_b if box.frame not in occupied], key=lambda box: box.frame)
                book.observations[id_a] = boxes_a
                items[index] = (id_a, boxes_a)
                absorbed.add(id_b)
                changed = True
                break
    for track_id in absorbed:
        book.observations.pop(track_id, None)


def _same_identity(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    gap_frames: int,
    fps: float,
) -> bool:
    """True only if two boxes are a plausible continuation of one object."""
    iou = _box_iou(a, b)
    ratio = _area_ratio(a, b)
    if ratio > 3.0:
        return False
    dt = gap_frames / max(fps, 1.0)
    dist = _center_dist(a, b)
    scale = min(_diag(a), _diag(b))
    if dt <= 0.12:
        if iou >= 0.25:
            return True
        return ratio <= 2.0 and dist <= 0.55 * max(scale, 1.0)
    if ratio > 2.0:
        return False
    if iou >= 0.4:
        return True
    return iou >= 0.25 and dist <= 0.35 * max(scale, 1.0)


def _area_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return max(area_a, area_b) / min(area_a, area_b)


def _center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _diag(box: tuple[float, float, float, float]) -> float:
    return max(1.0, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def stabilize_tracks(
    book: TrackBook,
    *,
    fps: float,
    max_gap_s: float,
    frame_stride: int,
    smooth_alpha: float,
    width: int,
    height: int,
) -> None:
    """Fill short dropouts and temporally smooth each track in place.

    Open-vocabulary detectors flicker: a billboard that stays in view can miss a
    few frames and ByteTrack then omits the box (the identity often survives).
    Linear interpolation closes those holes; a bidirectional EMA damps jitter.
    Gaps longer than ``max_gap_s`` are left empty so a real exit is not faked.
    Stride-sized holes are always filled so skipped detect frames still draw.
    """
    if fps <= 0:
        return
    stride_holes = max(0, frame_stride - 1)
    max_gap_frames = max(stride_holes, int(round(max(0.0, max_gap_s) * fps)))
    for track_id, boxes in list(book.observations.items()):
        filled = interpolate_gaps(boxes, max_gap_frames=max_gap_frames, fps=fps)
        filled = smooth_segments(filled, alpha=smooth_alpha)
        book.observations[track_id] = [clip_obs(box, width, height) for box in filled]


def interpolate_gaps(boxes: list[BoxObs], *, max_gap_frames: int, fps: float) -> list[BoxObs]:
    if len(boxes) <= 1 or max_gap_frames <= 0:
        return list(boxes)
    ordered = sorted(boxes, key=lambda box: box.frame)
    out: list[BoxObs] = [ordered[0]]
    for prev, nxt in zip(ordered, ordered[1:]):
        missing = nxt.frame - prev.frame - 1
        if 0 < missing <= max_gap_frames and _same_identity(
            prev.xyxy, nxt.xyxy, gap_frames=missing + 1, fps=fps
        ):
            span = nxt.frame - prev.frame
            for frame in range(prev.frame + 1, nxt.frame):
                weight = (frame - prev.frame) / span
                out.append(
                    BoxObs(
                        frame=frame,
                        t=frame / fps,
                        xyxy=_lerp_xyxy(prev.xyxy, nxt.xyxy, weight),
                        conf=prev.conf + (nxt.conf - prev.conf) * weight,
                        label=prev.label,
                        interpolated=True,
                    )
                )
        out.append(nxt)
    return out


def smooth_segments(boxes: list[BoxObs], *, alpha: float) -> list[BoxObs]:
    if alpha >= 1.0 or len(boxes) < 3:
        return list(boxes)
    alpha = min(1.0, max(0.05, alpha))
    out: list[BoxObs] = []
    for segment in _contiguous_segments(boxes):
        out.extend(_smooth_segment(segment, alpha=alpha))
    return out


def clip_obs(box: BoxObs, width: int, height: int) -> BoxObs:
    if width <= 1 or height <= 1:
        return box
    x1, y1, x2, y2 = box.xyxy
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 1.0), float(width))
    y2 = min(max(y2, 1.0), float(height))
    if x2 <= x1:
        x2 = min(x1 + 1.0, float(width))
    if y2 <= y1:
        y2 = min(y1 + 1.0, float(height))
    if box.xyxy == (x1, y1, x2, y2):
        return box
    return BoxObs(
        frame=box.frame,
        t=box.t,
        xyxy=(x1, y1, x2, y2),
        conf=box.conf,
        label=box.label,
        interpolated=box.interpolated,
    )


def _lerp_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    weight: float,
) -> tuple[float, float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * weight,
        a[1] + (b[1] - a[1]) * weight,
        a[2] + (b[2] - a[2]) * weight,
        a[3] + (b[3] - a[3]) * weight,
    )


def _contiguous_segments(boxes: list[BoxObs]) -> list[list[BoxObs]]:
    ordered = sorted(boxes, key=lambda box: box.frame)
    segments: list[list[BoxObs]] = []
    current: list[BoxObs] = [ordered[0]]
    for box in ordered[1:]:
        if box.frame == current[-1].frame + 1:
            current.append(box)
        else:
            segments.append(current)
            current = [box]
    segments.append(current)
    return segments


def _smooth_segment(boxes: list[BoxObs], *, alpha: float) -> list[BoxObs]:
    if len(boxes) < 3:
        return list(boxes)
    coords = [[box.xyxy[i] for box in boxes] for i in range(4)]
    smoothed = [_bidirectional_ema(series, alpha) for series in coords]
    out: list[BoxObs] = []
    for i, box in enumerate(boxes):
        out.append(
            BoxObs(
                frame=box.frame,
                t=box.t,
                xyxy=(smoothed[0][i], smoothed[1][i], smoothed[2][i], smoothed[3][i]),
                conf=box.conf,
                label=box.label,
                interpolated=box.interpolated,
            )
        )
    return out


def _bidirectional_ema(values: list[float], alpha: float) -> list[float]:
    forward = _ema(values, alpha)
    backward = _ema(list(reversed(values)), alpha)
    backward.reverse()
    return [(a + b) * 0.5 for a, b in zip(forward, backward)]


def _ema(values: list[float], alpha: float) -> list[float]:
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out
