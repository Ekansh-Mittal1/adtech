# Adtech

Two projects: capture glasses video on iPhone, then detect and track billboards on a Mac.

## GlassesCapture

Lightweight iOS app: stream the Ray-Ban Meta camera to the phone and save footage locally.

DAT cannot send the glasses camera to a Mac. This app is the phone side: frames come in over DAT and are written as `.mov` files.

### What it does

1. Connects through the **Meta AI** app (Developer Mode).
2. Starts a glasses camera session.
3. Records continuously. Ray-Ban Meta often pauses the camera around **1 minute**; the app saves that clip, then starts the next one. Files are also split every 5 minutes.
4. Stores clips in **Files → On My iPhone → GlassesCapture → Clips**.
5. Also copies each finished clip into **Photos**.

Keep the app in the foreground while recording. iOS will suspend it if you leave.

### Run it

1. In Meta AI: Settings → App Info → tap **App version** five times → enable **Developer Mode**. Pair the glasses.
2. Open `GlassesCapture/GlassesCapture.xcodeproj` in Xcode.
3. Signing & Capabilities: your Apple Team, bundle id `com.adtech.GlassesCapture`.
4. Run on a physical iPhone (not Simulator) with Meta AI installed.
5. Tap **Connect glasses**, approve in Meta AI, then **Start capture**.

DAT 0.9.0 expects Meta AI **V282** and Ray-Ban Meta firmware **V126**.

## BillboardDetect

Python CLI on the Mac: open-vocabulary detection (YOLO-World by default, Grounding DINO optional) plus ByteTrack. Input is a `.mov` / `.mp4` or a folder of clips. Output is an annotated MP4, a JSON track file, and a CSV of ad appearances. Pass `--ocr` to read text from lasting tracks (reuses `--from-detections` or `--from-tracks` so you do not have to run the detector again).

```bash
cd BillboardDetect
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m billboard_detect path/to/glasses-clip.mov
```

See [BillboardDetect/README.md](BillboardDetect/README.md) for FFmpeg (HEVC in, H.264 MP4 out) and extra options.
