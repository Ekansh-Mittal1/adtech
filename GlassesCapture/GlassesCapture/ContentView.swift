import SwiftUI

struct ContentView: View {
  @Bindable var controller: StreamController

  var body: some View {
    NavigationStack {
      VStack(spacing: 20) {
        statusRow("Meta AI", controller.registrationLabel)
        statusRow("Session", String(describing: controller.sessionState))
        statusRow("Stream", String(describing: controller.streamState))
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in
          VStack(spacing: 20) {
            statusRow("Recording", controller.isRecording ? "saving clips" : (controller.wantsCapture ? "starting next clip" : "idle"))
            statusRow("This clip", controller.recordingElapsedLabel)
          }
        }
        statusRow("Clips on phone", "\(controller.clipCount)")
        if let note = controller.statusNote {
          Text(note)
            .font(.subheadline)
            .multilineTextAlignment(.center)
            .padding(10)
            .frame(maxWidth: .infinity)
            .background(.green.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
        } else if !controller.lastClipName.isEmpty {
          Text(controller.lastClipName)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }

        Spacer()

        if controller.isRegistered {
          Button(controller.isCapturing ? "Stop" : "Start capture") {
            Task {
              if controller.isCapturing {
                await controller.stopCapture()
              } else {
                await controller.startCapture()
              }
            }
          }
          .buttonStyle(.borderedProminent)
          .tint(controller.isCapturing ? .red : .green)

          Button("Disconnect glasses", role: .destructive) {
            controller.disconnect()
          }
        } else {
          Button(controller.registrationState == .registering ? "Opening Meta AI…" : "Connect glasses") {
            controller.connect()
          }
          .buttonStyle(.borderedProminent)
          .disabled(controller.registrationState == .registering)
        }

        Text("The glasses camera often pauses around 1 minute. This app saves that clip, then starts the next one while the LED is still on. Files → On My iPhone → GlassesCapture, and Photos. Swiping the app away stops capture.")
          .font(.footnote)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }
      .padding()
      .navigationTitle("Glasses Capture")
      .alert("Camera access", isPresented: $controller.needsCameraPermissionConfirm) {
        Button("Continue") {
          Task { await controller.confirmCameraPermission() }
        }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text("Meta AI will ask for camera permission, then this app can record from your glasses.")
      }
      .alert("Something went wrong", isPresented: Binding(
        get: { controller.errorMessage != nil },
        set: { if !$0 { controller.errorMessage = nil } }
      )) {
        Button("OK", role: .cancel) { controller.errorMessage = nil }
      } message: {
        Text(controller.errorMessage ?? "")
      }
    }
  }

  private func statusRow(_ label: String, _ value: String) -> some View {
    HStack {
      Text(label)
      Spacer()
      Text(value)
        .foregroundStyle(.secondary)
        .monospaced()
    }
  }
}
