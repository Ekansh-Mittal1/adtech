import Foundation
import MWDATCamera
import MWDATCore
import Observation
import os

@Observable
@MainActor
final class StreamController {
  var registrationState: RegistrationState
  var sessionState: DeviceSessionState = .idle
  var streamState: StreamState = .stopped
  var isRecording = false
  var wantsCapture = false
  var clipCount = 0
  var lastClipName = ""
  var lastClipDuration: TimeInterval?
  var statusNote: String?
  var errorMessage: String?
  var needsCameraPermissionConfirm = false

  private let wearables: WearablesInterface
  private let recorder = ClipRecorder()
  private var deviceSelector: AutoDeviceSelector?
  private var lastDeviceId: DeviceIdentifier?
  private var deviceSession: DeviceSession?
  private var camera: Camera?
  private var sessionTokens: [AnyListenerToken] = []
  private var streamTokens: [AnyListenerToken] = []
  private var registrationTask: Task<Void, Never>?
  private var devicesTask: Task<Void, Never>?
  private var stallWatchdog: Task<Void, Never>?
  private var lastFrameAt: Date?
  private var streamArmedAt: Date?
  private var isRestarting = false
  private static let logger = Logger(subsystem: "com.adtech.GlassesCapture", category: "Stream")

  var isRegistered: Bool { registrationState == .registered }
  var isStreaming: Bool { streamState == .streaming }
  var isCapturing: Bool { wantsCapture }

  var recordingElapsedLabel: String {
    guard wantsCapture else { return "idle" }
    guard let start = recorder.recordingStartDate else { return "starting next clip…" }
    let seconds = max(0, Int(Date().timeIntervalSince(start)))
    return String(format: "%d:%02d", seconds / 60, seconds % 60)
  }

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
          self?.noteClip(url, duration: note.userInfo?["duration"] as? TimeInterval)
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
    do {
      if try await wearables.checkPermissionStatus(.camera) != .granted {
        needsCameraPermissionConfirm = true
        return
      }
    } catch {
      errorMessage = error.localizedDescription
      return
    }
    wantsCapture = true
    startDeviceWatch()
    ensureWatchdog()
    await startNewSession(reason: nil)
  }

  func confirmCameraPermission() async {
    needsCameraPermissionConfirm = false
    do {
      guard try await wearables.requestPermission(.camera) == .granted else {
        errorMessage = "Camera permission denied."
        return
      }
      wantsCapture = true
      startDeviceWatch()
      ensureWatchdog()
      await startNewSession(reason: nil)
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func stopCapture() async {
    wantsCapture = false
    isRestarting = false
    stallWatchdog?.cancel()
    stallWatchdog = nil
    recorder.stopAcceptingFrames()
    _ = await recorder.finishCurrentClip()
    isRecording = false
    BackgroundKeepAlive.stop()
    teardownSession()
    statusNote = nil
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

  private func startNewSession(reason: String?) async {
    guard wantsCapture, !isRestarting else { return }
    isRestarting = true
    if let reason { statusNote = reason }

    recorder.stopAcceptingFrames()
    _ = await recorder.finishCurrentClip()
    isRecording = false
    lastFrameAt = nil

    teardownSession()
    try? await Task.sleep(nanoseconds: 2_000_000_000)

    var started = false
    for attempt in 1...8 {
      guard wantsCapture else {
        isRestarting = false
        return
      }
      statusNote = attempt == 1
        ? (reason ?? "Connecting to glasses…")
        : "Waiting for glasses (try \(attempt))…"
      guard let selector = await makeSelector(timeout: 12) else {
        Self.logger.error("No eligible device after wait (attempt \(attempt))")
        continue
      }
      do {
        let session = try wearables.createSession(deviceSelector: selector)
        lastDeviceId = session.deviceId
        deviceSession = session
        sessionTokens.append(session.statePublisher.listen { [weak self] state in
          Task { @MainActor in self?.handleSessionState(state) }
        })
        sessionTokens.append(session.errorPublisher.listen { [weak self] error in
          Task { @MainActor in
            guard let self, !self.isRestarting else { return }
            self.errorMessage = error.localizedDescription
          }
        })
        sessionState = .starting
        try session.start()
        let ready = await waitUntil({ self.sessionState == .started && self.deviceSession != nil }, timeout: 8)
        if ready, let session = deviceSession {
          lastDeviceId = session.deviceId
          beginStream(on: session)
          started = camera != nil
        }
        if started { break }
        Self.logger.error("Session attempt \(attempt) did not start the camera")
      } catch let error as DeviceSessionError {
        Self.logger.error("createSession/start failed: \(error.description)")
        statusNote = error.description
        teardownSession()
      } catch {
        Self.logger.error("Session failed: \(error.localizedDescription)")
        teardownSession()
      }
      try? await Task.sleep(nanoseconds: 1_500_000_000)
    }

    if !started {
      isRestarting = false
      if wantsCapture {
        statusNote = "Could not restart the glasses camera. Tap Start capture to try again."
      }
    }
  }

  private func handleSessionState(_ state: DeviceSessionState) {
    sessionState = state
    if state == .stopped, !isRestarting {
      streamTokens.removeAll()
      camera = nil
      sessionTokens.removeAll()
      deviceSession = nil
      streamState = .stopped
      isRecording = false
      recorder.stopAcceptingFrames()
      Task { _ = await recorder.finishCurrentClip() }
      BackgroundKeepAlive.stop()
      if wantsCapture {
        Task { await self.startNewSession(reason: "Glasses session ended. Starting the next clip…") }
      }
    }
  }

  private func beginStream(on session: DeviceSession) {
    guard camera == nil else { return }
    let config = StreamConfiguration(
      videoCodec: .hvc1,
      resolution: .low,
      frameRate: 24
    )
    do {
      guard let newCamera = try session.addCamera(config: config) else {
        Self.logger.error("addCamera returned nil — session may not be started")
        return
      }
      camera = newCamera
      let stream = newCamera.stream
      streamTokens.append(stream.statePublisher.listen { [weak self] state in
        Task { @MainActor in self?.handleStreamState(state) }
      })
      streamTokens.append(stream.videoFramePublisher.listen { [weak self] frame in
        self?.recorder.append(frame.sampleBuffer)
        Task { @MainActor in self?.didReceiveFrame() }
      })
      streamTokens.append(stream.errorPublisher.listen { [weak self] error in
        Task { @MainActor in self?.handleStreamError(error) }
      })
      streamArmedAt = Date()
      lastFrameAt = nil
      recorder.startAcceptingFrames()
      stream.start()
    } catch {
      Self.logger.error("addCamera failed: \(error.localizedDescription)")
    }
  }

  private func didReceiveFrame() {
    lastFrameAt = Date()
    isRestarting = false
    isRecording = recorder.isRecording
  }

  private func handleStreamState(_ state: StreamState) {
    streamState = state
    if state == .streaming {
      BackgroundKeepAlive.start()
      recorder.startAcceptingFrames()
    } else if state == .paused {
      recorder.stopAcceptingFrames()
      Task { _ = await recorder.finishCurrentClip() }
      isRecording = false
      statusNote = "Glasses paused. Recording will resume when they wake."
    }
  }

  private func handleStreamError(_ error: StreamError) {
    switch error {
    case .hingesClosed, .thermalCritical, .thermalEmergency, .peakPowerShutdown, .batteryCritical:
      wantsCapture = false
      errorMessage = error.localizedDescription
    default:
      break
    }
  }

  private func teardownSession() {
    streamTokens.removeAll()
    camera?.stop()
    camera = nil
    streamState = .stopped
    sessionTokens.removeAll()
    deviceSession?.stop()
    deviceSession = nil
    sessionState = .stopped
  }

  private func ensureWatchdog() {
    guard stallWatchdog == nil else { return }
    stallWatchdog = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 500_000_000)
        guard let self, !Task.isCancelled else { return }
        await self.checkForStall()
      }
    }
  }

  private func checkForStall() async {
    guard wantsCapture else { return }
    if isRestarting {
      if let armed = streamArmedAt, Date().timeIntervalSince(armed) > 10 {
        Self.logger.info("New session produced no frames — trying another session")
        isRestarting = false
        await startNewSession(reason: "Glasses camera did not resume. Retrying…")
      }
      return
    }
    if let last = lastFrameAt, Date().timeIntervalSince(last) > 2.5 {
      Self.logger.info("Frames stalled — opening a new glasses session")
      await startNewSession(reason: "Saved clip. Starting the next one…")
    }
  }

  private func startDeviceWatch() {
    if deviceSelector == nil {
      deviceSelector = AutoDeviceSelector(wearables: wearables)
    }
    guard devicesTask == nil else { return }
    devicesTask = Task { [weak self] in
      guard let self else { return }
      for await devices in self.wearables.devicesStream() {
        if self.lastDeviceId == nil, let id = devices.first {
          self.lastDeviceId = id
        }
      }
    }
  }

  private func connectedDeviceId() -> DeviceIdentifier? {
    if let id = lastDeviceId, let device = wearables.deviceForIdentifier(id), device.linkState == .connected {
      return id
    }
    if let id = deviceSelector?.activeDevice,
      let device = wearables.deviceForIdentifier(id),
      device.linkState == .connected
    {
      return id
    }
    for id in wearables.devices {
      if let device = wearables.deviceForIdentifier(id), device.linkState == .connected {
        return id
      }
    }
    return wearables.devices.first
  }

  private func makeSelector(timeout: TimeInterval) async -> (any DeviceSelector)? {
    if deviceSelector == nil {
      deviceSelector = AutoDeviceSelector(wearables: wearables)
    }
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if let id = connectedDeviceId() {
        lastDeviceId = id
        return SpecificDeviceSelector(device: id)
      }
      try? await Task.sleep(nanoseconds: 200_000_000)
    }
    if let id = connectedDeviceId() {
      lastDeviceId = id
      return SpecificDeviceSelector(device: id)
    }
    return nil
  }

  private func waitUntil(_ condition: @escaping () -> Bool, timeout: TimeInterval) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if condition() { return true }
      try? await Task.sleep(nanoseconds: 100_000_000)
    }
    return condition()
  }

  private func noteClip(_ url: URL, duration: TimeInterval?) {
    clipCount += 1
    lastClipName = url.lastPathComponent
    lastClipDuration = duration
    if let duration {
      let seconds = max(0, Int(duration.rounded()))
      let length = "\(seconds / 60):\(String(format: "%02d", seconds % 60))"
      statusNote = wantsCapture
        ? "Saved \(url.lastPathComponent) (\(length)). Continuing…"
        : "Saved \(url.lastPathComponent) (\(length))."
    } else {
      statusNote = "Saved \(url.lastPathComponent)."
    }
  }
}
