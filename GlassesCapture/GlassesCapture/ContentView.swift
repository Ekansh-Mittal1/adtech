import SwiftUI

struct ContentView: View {
  @Bindable var controller: StreamController

  var body: some View {
    NavigationStack {
      VStack(spacing: 20) {
        statusRow("Meta AI", controller.registrationLabel)
        statusRow("Session", String(describing: controller.sessionState))
        statusRow("Stream", String(describing: controller.streamState))
        statusRow("Recording", controller.isRecording ? "saving clips" : "idle")
        statusRow("Clips on phone", "\(controller.clipCount)")
        if !controller.lastClipName.isEmpty {
          Text(controller.lastClipName)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }

        Spacer()

        if controller.isRegistered {
          Button(controller.isStreaming ? "Stop" : "Start capture") {
            Task {
              if controller.isStreaming {
                await controller.stopCapture()
              } else {
                await controller.startCapture()
              }
            }
          }
          .buttonStyle(.borderedProminent)
          .tint(controller.isStreaming ? .red : .green)

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

        Text("Clips are stored in Files → On My iPhone → GlassesCapture, and copied to Photos. You can lock the phone or switch apps while capture is running. Swiping the app away stops it — iOS will not restart DAT.")
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
