import AVFoundation
import CoreMedia
import os

/// Writes HEVC frames in passthrough so the MOV is Photos-compatible.
final class VideoCaptureHandler: Sendable {
  private static let fallbackFrameRate: Int32 = 24
  private static let logger = Logger(subsystem: "com.adtech.GlassesCapture", category: "VideoCapture")

  private struct State {
    var videoInput: AVAssetWriterInput
    weak var assetWriter: AVAssetWriter?
    var streamStartTimestamp: CMTime?
    var lastOutputDTS: CMTime?
    var isCapturing: Bool = false
  }

  private let state: OSAllocatedUnfairLock<State>

  init(writer: AVAssetWriter, sourceFormatHint: CMFormatDescription?) {
    let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: sourceFormatHint)
    videoInput.expectsMediaDataInRealTime = true
    state = OSAllocatedUnfairLock(uncheckedState: State(videoInput: videoInput, assetWriter: writer))
    if writer.canAdd(videoInput) {
      writer.add(videoInput)
    } else {
      Self.logger.error("Could not add video input")
    }
  }

  func start() {
    state.withLockUnchecked { $0.isCapturing = true }
  }

  func stop() {
    state.withLockUnchecked {
      $0.isCapturing = false
      $0.videoInput.markAsFinished()
    }
  }

  func appendVideoFrame(_ sampleBuffer: CMSampleBuffer) {
    guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
      CMFormatDescriptionGetMediaType(formatDescription) == kCMMediaType_Video
    else { return }

    let sourcePTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    let sourceDTS = CMSampleBufferGetDecodeTimeStamp(sampleBuffer)
    let originalDuration = CMSampleBufferGetDuration(sampleBuffer)
    let frameDuration =
      (originalDuration.isValid && originalDuration > .zero)
      ? originalDuration
      : CMTimeMake(value: 1, timescale: Self.fallbackFrameRate)
    let isSync = sampleBuffer.isHEVCKeyframe()

    state.withLockUnchecked { state in
      guard state.isCapturing, state.assetWriter?.status != .failed, state.videoInput.isReadyForMoreMediaData else {
        return
      }

      let timelineStart = state.streamStartTimestamp ?? sourcePTS
      if state.streamStartTimestamp == nil {
        state.streamStartTimestamp = sourcePTS
      }

      var dts = sourceDTS.isValid ? CMTimeSubtract(sourceDTS, timelineStart) : CMTimeSubtract(sourcePTS, timelineStart)
      if let lastDTS = state.lastOutputDTS {
        let floor = CMTimeAdd(lastDTS, frameDuration)
        if dts < floor { dts = floor }
      }
      var pts = CMTimeSubtract(sourcePTS, timelineStart)
      if pts < dts { pts = dts }

      var timingInfo = CMSampleTimingInfo(duration: frameDuration, presentationTimeStamp: pts, decodeTimeStamp: dts)
      var adjusted: CMSampleBuffer?
      guard
        CMSampleBufferCreateCopyWithNewTiming(
          allocator: kCFAllocatorDefault,
          sampleBuffer: sampleBuffer,
          sampleTimingEntryCount: 1,
          sampleTimingArray: &timingInfo,
          sampleBufferOut: &adjusted
        ) == noErr, let adjusted
      else { return }

      if let attachments = CMSampleBufferGetSampleAttachmentsArray(adjusted, createIfNecessary: true) as? [NSMutableDictionary],
        let dict = attachments.first
      {
        dict[kCMSampleAttachmentKey_NotSync] = !isSync
        dict[kCMSampleAttachmentKey_DependsOnOthers] = !isSync
      }

      if state.videoInput.append(adjusted) {
        state.lastOutputDTS = dts
      }
    }
  }
}

extension CMSampleBuffer {
  func isHEVCKeyframe() -> Bool {
    guard let dataBuffer = CMSampleBufferGetDataBuffer(self) else { return false }
    let totalLength = CMBlockBufferGetDataLength(dataBuffer)
    var offset = 0
    while offset + 4 < totalLength {
      var nalLengthBE: UInt32 = 0
      guard CMBlockBufferCopyDataBytes(dataBuffer, atOffset: offset, dataLength: 4, destination: &nalLengthBE) == kCMBlockBufferNoErr
      else { break }
      let nalLength = Int(UInt32(bigEndian: nalLengthBE))
      offset += 4
      guard nalLength > 0, offset + nalLength <= totalLength else { break }
      var nalHeader: UInt8 = 0
      guard CMBlockBufferCopyDataBytes(dataBuffer, atOffset: offset, dataLength: 1, destination: &nalHeader) == kCMBlockBufferNoErr
      else { break }
      let nalUnitType = (nalHeader >> 1) & 0x3F
      if nalUnitType >= 16 && nalUnitType <= 21 { return true }
      offset += nalLength
    }
    return false
  }
}
