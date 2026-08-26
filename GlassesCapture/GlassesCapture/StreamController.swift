import Foundation
import MWDATCamera
import MWDATCore
import Observation

@Observable
@MainActor
final class StreamController {
  var registrationState: RegistrationState
  var sessionState: DeviceSessionState = .idle
  var streamState: StreamState = .stopped
  var isRecording = false
  var clipCount = 0
  var lastClipName = ""
  var errorMessage: String?
  var needsCameraPermissionConfirm = false

  private let wearables: WearablesInterface
  private let deviceSelector: AutoDeviceSelector
  private let recorder = ClipRecorder()
  private var deviceSession: DeviceSession?
  private var camera: Camera?
  private var tokens: [AnyListenerToken] = []
  private var registrationTask: Task<Void, Never>?
  private var autoStartStream = false

  var isRegistered: Bool { registrationState == .registered }
  var isStreaming: Bool { streamState == .streaming }

  var registrationLabel: String {
    switch registrationState {
    case .unavailable: return "Unavailable"
    case .available: return "Available — tap Connect"
    case .registering: return "Registering…"
    case .registered: return "Registered"
    @unknown default: return "Unknown (\(registrationState.rawValue))"
    }
  }

  init(wearables: WearablesInterface = Wearables.shared) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    self.registrationState = wearables.registrationState
    clipCount = (try? FileManager.default.contentsOfDirectory(at: ClipRecorder.clipsDirectory, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "mov" }.count) ?? 0
    registrationTask = Task { [weak self] in
      guard let self else { return }
      for await state in wearables.registrationStateStream() {
        self.registrationState = state
      }
    }
    Task { [weak self] in
      for await note in NotificationCenter.default.notifications(named: .glassesClipSaved) {
        if let url = note.object as? URL {
          self?.noteClip(url)
        }
      }
    }
  }

  deinit {}

  func connect() {
    if registrationState == .registering {
      errorMessage = "Registration is already in progress. If Meta AI did not open, open it manually and check App connections → Developer mode apps."
      return
    }
    Task {
      do {
        try await wearables.startRegistration()
      } catch let error as RegistrationError {
        errorMessage = error.description
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  func disconnect() {
    Task { await stopCapture() }
    Task {
      do {
        try await wearables.startUnregistration()
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  func startCapture() async {
    guard isRegistered else {
      errorMessage = "Connect to Meta AI first."
      return
    }
    guard deviceSession == nil else { return }
    do {
      if try await wearables.checkPermissionStatus(.camera) != .granted {
        needsCameraPermissionConfirm = true
        return
      }
    } catch {
      errorMessage = error.localizedDescription
      return
    }
    beginSession()
  }

  func confirmCameraPermission() async {
    needsCameraPermissionConfirm = false
    do {
      guard try await wearables.requestPermission(.camera) == .granted else {
        errorMessage = "Camera permission denied."
        return
      }
      beginSession()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func stopCapture() async {
    autoStartStream = false
    recorder.stopAcceptingFrames()
    _ = await recorder.finishCurrentClip()
    isRecording = false
    BackgroundKeepAlive.stop()
    camera?.stop()
    deviceSession?.stop()
  }

  func handleOpenURL(_ url: URL) async {
    guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
      components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
    else { return }
    do {
      _ = try await Wearables.shared.handleUrl(url)
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func beginSession() {
    do {
      let session = try wearables.createSession(deviceSelector: deviceSelector)
      deviceSession = session
      autoStartStream = true
      tokens.append(session.statePublisher.listen { [weak self] state in
        Task { @MainActor in self?.handleSessionState(state) }
      })
      tokens.append(session.errorPublisher.listen { [weak self] error in
        Task { @MainActor in self?.errorMessage = error.localizedDescription }
      })
      sessionState = .starting
      try session.start()
    } catch {
      errorMessage = error.localizedDescription
      deviceSession = nil
    }
  }

  private func handleSessionState(_ state: DeviceSessionState) {
    sessionState = state
    if state == .started, autoStartStream, camera == nil, let session = deviceSession {
      beginStream(on: session)
    }
    if state == .stopped {
      recorder.stopAcceptingFrames()
      Task {
        _ = await recorder.finishCurrentClip()
      }
      tokens.removeAll()
      camera = nil
      deviceSession = nil
      streamState = .stopped
      isRecording = false
      BackgroundKeepAlive.stop()
    }
  }

  private func beginStream(on session: DeviceSession) {
    let config = StreamConfiguration(
      videoCodec: .hvc1,
      resolution: .low,
      frameRate: 24
    )
    do {
      guard let newCamera = try session.addCamera(config: config) else {
        errorMessage = "Could not start the glasses camera."
        return
      }
      camera = newCamera
      let stream = newCamera.stream
      tokens.append(stream.statePublisher.listen { [weak self] state in
        Task { @MainActor in
          self?.streamState = state
          if state == .streaming {
            BackgroundKeepAlive.start()
            self?.recorder.startAcceptingFrames()
            self?.isRecording = true
          }
        }
      })
      tokens.append(stream.videoFramePublisher.listen { [weak self] frame in
        self?.recorder.append(frame.sampleBuffer)
        Task { @MainActor in
          self?.isRecording = self?.recorder.isRecording ?? false
        }
      })
      tokens.append(stream.errorPublisher.listen { [weak self] error in
        Task { @MainActor in self?.errorMessage = error.localizedDescription }
      })
      streamState = .starting
      stream.start()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func noteClip(_ url: URL) {
    clipCount += 1
    lastClipName = url.lastPathComponent
  }
}
