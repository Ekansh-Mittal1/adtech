import AVFoundation
import UIKit

/// Keeps the process eligible to run after the user leaves the screen.
/// DAT will only keep delivering HEVC frames while iOS has not suspended us.
enum BackgroundKeepAlive {
  private static var recorder: AVAudioRecorder?

  static func start() {
    UIApplication.shared.isIdleTimerDisabled = true
    let session = AVAudioSession.sharedInstance()
    do {
      try session.setCategory(.playAndRecord, mode: .videoRecording, options: [.mixWithOthers, .defaultToSpeaker])
      try session.setActive(true)
    } catch {
      try? session.setCategory(.playback, options: [.mixWithOthers])
      try? session.setActive(true)
    }

    let url = FileManager.default.temporaryDirectory.appendingPathComponent("keepalive.caf")
    let settings: [String: Any] = [
      AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
      AVSampleRateKey: 8_000,
      AVNumberOfChannelsKey: 1,
      AVEncoderAudioQualityKey: AVAudioQuality.min.rawValue
    ]
    recorder = try? AVAudioRecorder(url: url, settings: settings)
    recorder?.record()
  }

  static func stop() {
    UIApplication.shared.isIdleTimerDisabled = false
    recorder?.stop()
    recorder = nil
    let keepalive = FileManager.default.temporaryDirectory.appendingPathComponent("keepalive.caf")
    try? FileManager.default.removeItem(at: keepalive)
    try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
  }
}
