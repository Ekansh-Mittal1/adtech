# Glasses Capture

Lightweight iOS app: stream the Ray-Ban Meta camera to the phone and save footage locally.

DAT cannot send the glasses camera to a Mac. This app is the phone side: frames come in over DAT and are written as `.mov` files.

## What it does

1. Connects through the **Meta AI** app (Developer Mode).
2. Starts a glasses camera session.
3. Records continuously into 5-minute clips.
4. Stores clips in **Files → On My iPhone → GlassesCapture → Clips**.
5. Also copies each finished clip into **Photos**.

Keep the app in the foreground while recording. iOS will suspend it if you leave.

## Run it

1. In Meta AI: Settings → App Info → tap **App version** five times → enable **Developer Mode**. Pair the glasses.
2. Open `GlassesCapture/GlassesCapture.xcodeproj` in Xcode.
3. Signing & Capabilities: your Apple Team, bundle id `com.adtech.GlassesCapture`.
4. Run on a physical iPhone (not Simulator) with Meta AI installed.
5. Tap **Connect glasses**, approve in Meta AI, then **Start capture**.

DAT 0.9.0 expects Meta AI **V282** and Ray-Ban Meta firmware **V126**.
