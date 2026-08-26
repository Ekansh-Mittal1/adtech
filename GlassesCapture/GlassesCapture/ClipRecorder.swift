import AVFoundation
import CoreMedia
import Foundation
import Photos
import os

/// Records the glasses HEVC stream to Documents/Clips and copies finished files into Photos.
final class ClipRecorder: Sendable {
  static let clipsDirectory: URL = {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("Clips", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }()

  private struct State {
    var assetWriter: AVAssetWriter?
    var videoHandler: VideoCaptureHandler?
    var outputURL: URL?
    var recordingStartTime: Date?
    var isRecording = false
    var shouldAcceptNewFrames = false
    var isRotating = false
  }

  private let state = OSAllocatedUnfairLock(uncheckedState: State())
  private static let logger = Logger(subsystem: "com.adtech.GlassesCapture", category: "ClipRecorder")
  private let maxSegment: TimeInterval

  var isRecording: Bool {
    state.withLockUnchecked { $0.isRecording }
  }

  var recordingStartDate: Date? {
    state.withLockUnchecked { $0.recordingStartTime }
  }

  init(maxSegment: TimeInterval = 5 * 60) {
    self.maxSegment = maxSegment
  }

  func startAcceptingFrames() {
    state.withLockUnchecked { $0.shouldAcceptNewFrames = true }
  }

  func stopAcceptingFrames() {
    state.withLockUnchecked { $0.shouldAcceptNewFrames = false }
  }

  func append(_ sampleBuffer: CMSampleBuffer) {
    let (recording, start, shouldAccept, rotating) = state.withLockUnchecked {
      ($0.isRecording, $0.recordingStartTime, $0.shouldAcceptNewFrames, $0.isRotating)
    }
    if rotating { return }

    if recording, let start, Date().timeIntervalSince(start) >= maxSegment, sampleBuffer.isHEVCKeyframe() {
      Task { await rotate() }
      return
    }

    if !recording {
      guard shouldAccept, sampleBuffer.isHEVCKeyframe() else { return }
      beginFile(with: sampleBuffer)
    }

    let handler = state.withLockUnchecked { $0.videoHandler }
    handler?.appendVideoFrame(sampleBuffer)
  }

  func finishCurrentClip() async -> URL? {
    await finalize()
  }

  private func rotate() async {
    let started = state.withLockUnchecked { state -> Bool in
      guard !state.isRotating else { return false }
      state.isRotating = true
      return true
    }
    guard started else { return }
    _ = await finalize()
    state.withLockUnchecked {
      $0.shouldAcceptNewFrames = true
      $0.isRotating = false
    }
  }

  private func beginFile(with sampleBuffer: CMSampleBuffer) {
    let (shouldAccept, already) = state.withLockUnchecked { ($0.shouldAcceptNewFrames, $0.isRecording) }
    guard shouldAccept, !already else { return }
    guard let format = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
    let dimensions = CMVideoFormatDescriptionGetDimensions(format)
    guard dimensions.width > 0, dimensions.height > 0 else { return }

    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd-HHmmss"
    let name = "glasses-\(formatter.string(from: Date())).mov"
    let url = Self.clipsDirectory.appendingPathComponent(name)

    do {
      let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
      let handler = VideoCaptureHandler(writer: writer, sourceFormatHint: format)
      guard writer.startWriting() else {
        Self.logger.error("Asset writer failed to start")
        return
      }
      writer.startSession(atSourceTime: .zero)
      let start = Date()
      state.withLockUnchecked { state in
        state.assetWriter = writer
        state.videoHandler = handler
        state.outputURL = url
        state.recordingStartTime = start
        state.isRecording = true
      }
      handler.start()
    } catch {
      Self.logger.error("Failed to create writer: \(error.localizedDescription)")
    }
  }

  private func finalize() async -> URL? {
    let (shouldStop, url, handler, writer) = state.withLockUnchecked { state -> (Bool, URL?, VideoCaptureHandler?, AVAssetWriter?) in
      guard state.isRecording else { return (false, nil, nil, nil) }
      state.isRecording = false
      return (true, state.outputURL, state.videoHandler, state.assetWriter)
    }
    guard shouldStop else { return nil }
    handler?.stop()
    guard let writer, let url else { return nil }
    await writer.finishWriting()
    let ok = writer.status == .completed
    state.withLockUnchecked { state in
      guard state.assetWriter === writer else { return }
      state.assetWriter = nil
      state.videoHandler = nil
      state.outputURL = nil
      state.recordingStartTime = nil
    }
    guard ok else {
      Self.logger.error("Writer did not finish (status \(writer.status.rawValue))")
      try? FileManager.default.removeItem(at: url)
      return nil
    }
    await Self.saveToPhotos(url)
    NotificationCenter.default.post(name: .glassesClipSaved, object: url)
    return url
  }

  private static func saveToPhotos(_ url: URL) async {
    let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
    guard status == .authorized || status == .limited else { return }
    do {
      try await PHPhotoLibrary.shared().performChanges {
        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
      }
    } catch {
      Self.logger.error("Photos save failed: \(error.localizedDescription)")
    }
  }
}

extension Notification.Name {
  static let glassesClipSaved = Notification.Name("glassesClipSaved")
}
