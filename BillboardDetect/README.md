# Billboard Detect

Python CLI that finds and tracks billboards / advertisements in video. Point it at a GlassesCapture `.mov` (or any clip). It writes an annotated MP4 plus JSON/CSV of each ad appearance.

Capture stays in [GlassesCapture](../GlassesCapture). This project runs on a Mac (Apple Silicon `mps`, NVIDIA `cuda`, or CPU).

## Install

```bash
cd BillboardDetect
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

YOLO-World weights (`yolov8s-worldv2.pt`) download on first run.

Grounding DINO is optional and heavier:

```bash
pip install -e '.[dino]'
```

OCR (RapidOCR) is optional:

```bash
pip install -e '.[ocr]'
```

A local vision model (Qwen2.5-VL via MLX, Apple Silicon) is better on logos and stylized type:

```bash
pip install -e '.[ocr-vlm]'
```

### FFmpeg

Required. The CLI uses the `ffmpeg` / `ffprobe` on your PATH to decode GlassesCapture HEVC `.mov` files and to write a playable H.264 MP4 (`yuv420p`, `+faststart`).

```bash
brew install ffmpeg
```

## Run

```bash
python -m billboard_detect path/to/glasses-clip.mov
python -m billboard_detect clips/
python -m billboard_detect clip.mov --detector grounding-dino --conf 0.05 0.1 0.25
python -m billboard_detect clip.mov --smooth-alpha 0.25
python -m billboard_detect clip.mov --no-smooth --max-gap 0
python -m billboard_detect clip.mov --ocr
python -m billboard_detect clip.mov --from-tracks output/clip/v001/clip_tracks.json --ocr
python -m billboard_detect clip.mov --from-tracks output/clip/v001/clip_tracks.json --ocr-engine vlm --no-annotate
```

Defaults: detector `yolo-world`, `--conf 0.25`, `--imgsz 1280`, `--frame-stride 2`, `--max-gap 1.0`, `--smooth`, output `./output/<stem>/v001/` (a new `vNNN` folder every run; older runs are left alone).

`--frame-stride` skips detection on some frames for speed. The annotated video still contains every source frame.

Detectors flicker even when a billboard stays in view, so ByteTrack would drop the box until the next hit. After tracking, gaps shorter than `--max-gap` seconds are linearly interpolated (stride holes are always filled). `--smooth` then applies a bidirectional EMA to the box corners (`--smooth-alpha` controls strength, default `0.4`). `--no-smooth` skips the EMA; `--max-gap 0 --no-smooth` restores the raw tracker output.

Pass several `--conf` values to export each threshold in one go. The model runs **once** at the lowest value, then ByteTrack / MP4 / JSON / CSV are produced per threshold (tracking is not a simple subset — identities can change when boxes drop out). Raw boxes are saved as `*_detections.json` so you can sweep again without inference:

```bash
python -m billboard_detect clip.mov --conf 0.05 0.1 0.25
python -m billboard_detect clip.mov --from-detections output/clip/v001/clip_detections.json --conf 0.08 0.2
```

`--from-detections` replays raw boxes (re-runs ByteTrack). `--from-tracks` reuses the already-tracked boxes, including which frames were interpolated.

## OCR

`--ocr` reads text from lasting tracks. It ignores interpolated boxes, keeps tracks that last at least 0.05 s (or have 2+ real boxes), and sends **one** tight real crop per track. `--ocr-engine vlm` classifies that single crop with Qwen2.5-VL (`generate` gets one image path — never a list of boxes). The overlay headline is drawn only while later boxes still match that crop. Street signs and storefront logos are not billboards; truck-side ads and vehicle wraps are. First VLM run downloads ~6 GB of weights.

Crops come from the **source video**, never the annotated MP4.

```bash
pip install -e '.[ocr]'
python -m billboard_detect clip.mov --ocr
python -m billboard_detect clip.mov --from-tracks output/clip/v001/clip_tracks.json --ocr
python -m billboard_detect clip.mov --from-detections output/clip/v001/clip_detections.json --ocr

pip install -e '.[ocr-vlm]'
python -m billboard_detect clip.mov --from-tracks output/clip/v001/clip_tracks.json --ocr-engine vlm --no-annotate
```

`--from-tracks` writes a **new** `output/<stem>/vNNN/` (or `--output/<stem>/vNNN/`). It never overwrites the folder the tracks came from. It also re-renders the annotated MP4.

## Output

For `glasses-20260826-183000.mov` the first run writes `output/glasses-20260826-183000/v001/`:

| File | Contents |
| --- | --- |
| `…_detections.json` | Every box the model returned (at the lowest `--conf`); used by `--from-detections` |
| `…_annotated.mp4` | H.264 MP4 with every tracked box. Overlay labels are `#id` plus the VLM headline, capped at 32 characters. |
| `…_annotated_ads.mp4` | Same overlay, but only tracks the VLM (or OCR) marked as ads. Written whenever `…_ocr.json` exists. |
| `…_tracks.json` | Per-track timeline: id, label, `t_start` / `t_end`, confidence, per-frame boxes (interpolated frames are flagged) |
| `…_appearances.csv` | One row per track (no per-frame boxes) |
| `…_ocr.json` / `…_ocr.csv` | Track-level VLM/OCR (`is_ad`, `headline`, `full_copy`). Non-billboards have empty copy. |

With more than one `--conf`, annotated MP4 / JSON / CSV go in `output/<stem>/vNNN/conf_<threshold>/`.

Open-vocabulary classes default to `billboard`, `digital billboard`, `outdoor billboard`, `LED billboard`, `advertisement`, `poster`, `banner`, `truck advertisement`, `truck-side advertisement`, `vehicle wrap`. Overlapping hits are NMS'd (the tighter, higher-confidence box is kept — boxes are never unioned). Change queries with `--classes` if a scene uses different wording.

YOLO-World is the default because it is fast enough for video. `--detector grounding-dino` is slower and often better on unusual or distant ads.
