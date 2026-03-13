import ExpoModulesCore
import AVFoundation
import CoreVideo

public class WebGPUCameraModule: Module {
  private var captureSession: AVCaptureSession?
  private var dataOutput: AVCaptureVideoDataOutput?
  private var frameDelegate: FrameDelegate?
  private let sessionQueue = DispatchQueue(label: "webgpu-camera-session")
  private let frameQueue = DispatchQueue(label: "webgpu-camera-frame", qos: .userInteractive)

  // --- Dawn compute pipeline ---
  var dawnBridge: DawnPipelineBridge?
  private var computeSetup = false

  // --- Recorder (Spike 4) ---
  private var assetWriter: AVAssetWriter?
  private var writerInput: AVAssetWriterInput?
  private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
  private var isRecording = false
  private var recordedFrameCount: Int64 = 0
  private var recordingOutputPath: String = ""

  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, width: Int, height: Int, fps: Int) in
      self.startCapture(deviceId: deviceId, width: width, height: height, fps: fps)
    }

    Function("stopCameraPreview") {
      self.stopCapture()
    }

    Function("getCurrentFrameHandle") { () -> Int in
      return Int(getCurrentFrameHandle())
    }

    Function("getCurrentFramePixels") { () -> Data in
      let pixels = getCurrentFramePixels()
      return Data(pixels)
    }

    Function("getFrameDimensions") { () -> [String: Any] in
      let dims = getFrameDimensions()
      return ["width": dims.width, "height": dims.height, "bytesPerRow": dims.bytesPerRow]
    }

    Function("getFrameCounter") { () -> Int in
      return Int(getFrameCounter())
    }

    Function("startTestRecorder") { (outputPath: String, width: Int, height: Int) -> Int in
      return self.startRecorder(outputPath: outputPath, width: width, height: height)
    }

    Function("stopTestRecorder") { () -> String in
      return self.stopRecorder()
    }

    Function("appendFrameToRecorder") { (pixels: Data, width: Int, height: Int) in
      self.appendFrameToRecorder(pixels: pixels, width: width, height: height)
    }

    Function("getThermalState") { () -> String in
      let state = ProcessInfo.processInfo.thermalState
      switch state {
      case .nominal: return "nominal"
      case .fair: return "fair"
      case .serious: return "serious"
      case .critical: return "critical"
      @unknown default: return "nominal"
      }
    }

    // --- Dawn compute pipeline ---

    Function("setupComputePipeline") { (wgslCode: String, width: Int, height: Int) -> Bool in
      let bridge = DawnPipelineBridge()
      let ok = bridge.setup(withWGSL: wgslCode, width: Int32(width), height: Int32(height))
      if ok {
        self.dawnBridge = bridge
        self.computeSetup = true
        // Install JSI bindings so JS can call __webgpuCamera_nextImage()
        if let runtime = try? self.appContext?.runtime {
          bridge.installJSIBindings(runtime)
        }
        print("[WebGPUCamera] Compute pipeline setup OK: \(width)x\(height)")
      } else {
        print("[WebGPUCamera] Compute pipeline setup FAILED")
      }
      return ok
    }

    Function("cleanupComputePipeline") {
      self.dawnBridge?.cleanup()
      self.dawnBridge = nil
      self.computeSetup = false
      print("[WebGPUCamera] Compute pipeline cleaned up")
    }

    Function("isComputeReady") { () -> Bool in
      return self.computeSetup
    }
  }

  private func startCapture(deviceId: String, width: Int, height: Int, fps: Int) {
    sessionQueue.async {
      let session = AVCaptureSession()
      session.sessionPreset = .hd1920x1080

      // Find camera device
      let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back
      guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
        print("[WebGPUCamera] No camera found for position: \(position)")
        return
      }

      do {
        let input = try AVCaptureDeviceInput(device: camera)
        if session.canAddInput(input) {
          session.addInput(input)
        }

        // Configure requested frame rate
        try camera.lockForConfiguration()
        let targetFPS = Double(fps)
        var bestFormat: AVCaptureDevice.Format?
        var bestRange: AVFrameRateRange?

        for format in camera.formats {
          let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
          if dims.width == Int32(width) && dims.height == Int32(height) {
            for range in format.videoSupportedFrameRateRanges {
              if range.maxFrameRate >= targetFPS {
                if bestRange == nil || range.maxFrameRate > bestRange!.maxFrameRate {
                  bestFormat = format
                  bestRange = range
                }
              }
            }
          }
        }

        if let format = bestFormat, let range = bestRange {
          camera.activeFormat = format
          camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(min(targetFPS, range.maxFrameRate)))
          camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(min(targetFPS, range.maxFrameRate)))
          print("[WebGPUCamera] Configured \(min(targetFPS, range.maxFrameRate))fps")
        } else {
          print("[WebGPUCamera] \(fps)fps not available at \(width)x\(height), using default")
        }
        camera.unlockForConfiguration()
      } catch {
        print("[WebGPUCamera] Failed to create camera input: \(error)")
        return
      }

      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
      output.alwaysDiscardsLateVideoFrames = true

      // Retain delegate as instance property to prevent deallocation
      let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
      self.frameDelegate = delegate
      output.setSampleBufferDelegate(delegate, queue: self.frameQueue)

      if session.canAddOutput(output) {
        session.addOutput(output)
      }

      // Update frame dimensions in Rust
      setFrameDimensions(width: UInt32(width), height: UInt32(height), bytesPerRow: UInt32(width * 4))

      session.startRunning()
      self.captureSession = session
      self.dataOutput = output
      print("[WebGPUCamera] Camera started: \(width)x\(height)")
    }
  }

  private func stopCapture() {
    sessionQueue.async {
      self.captureSession?.stopRunning()
      self.captureSession = nil
      self.dataOutput = nil
      self.frameDelegate = nil
      print("[WebGPUCamera] Camera stopped")
    }
  }

  // MARK: - Recorder

  private func startRecorder(outputPath: String, width: Int, height: Int) -> Int {
    let url = URL(fileURLWithPath: outputPath)
    try? FileManager.default.removeItem(at: url)

    do {
      let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

      let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: 10_000_000,
          AVVideoExpectedSourceFrameRateKey: 30,
        ]
      ]

      let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
      input.expectsMediaDataInRealTime = true

      let sourcePixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ]

      let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: sourcePixelBufferAttributes
      )

      if writer.canAdd(input) {
        writer.add(input)
      }

      writer.startWriting()
      writer.startSession(atSourceTime: .zero)

      self.assetWriter = writer
      self.writerInput = input
      self.pixelBufferAdaptor = adaptor
      self.isRecording = true
      self.recordedFrameCount = 0
      self.recordingOutputPath = outputPath

      print("[WebGPUCamera] Recorder started: \(outputPath)")
      return 0 // Readback path
    } catch {
      print("[WebGPUCamera] Recorder setup failed: \(error)")
      return 0
    }
  }

  private func stopRecorder() -> String {
    guard let writer = assetWriter, isRecording else { return "" }

    isRecording = false
    writerInput?.markAsFinished()

    let semaphore = DispatchSemaphore(value: 0)
    let outputPath = recordingOutputPath

    writer.finishWriting {
      print("[WebGPUCamera] Recording finished: \(self.recordedFrameCount) frames, path: \(outputPath)")
      semaphore.signal()
    }
    semaphore.wait()

    assetWriter = nil
    writerInput = nil
    pixelBufferAdaptor = nil

    return outputPath
  }

  func appendFrameToRecorder(pixels: Data, width: Int, height: Int) {
    guard isRecording,
          let adaptor = pixelBufferAdaptor,
          let input = writerInput,
          input.isReadyForMoreMediaData else { return }

    guard let pool = adaptor.pixelBufferPool else { return }

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
      pixels.copyBytes(to: baseAddress.assumingMemoryBound(to: UInt8.self), count: min(pixels.count, width * height * 4))
    }

    let frameTime = CMTime(value: recordedFrameCount, timescale: 30)
    adaptor.append(buffer, withPresentationTime: frameTime)
    recordedFrameCount += 1
  }
}

private class FrameDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  let width: UInt32
  let height: UInt32
  weak var module: WebGPUCameraModule?

  init(width: UInt32, height: UInt32, module: WebGPUCameraModule) {
    self.width = width
    self.height = height
    self.module = module
  }

  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    // Run Dawn compute pipeline on the raw CVPixelBuffer (zero-copy via IOSurface)
    module?.dawnBridge?.processFrame(pixelBuffer)

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let dataSize = bytesPerRow * Int(height)

    // Capture IOSurface handle FIRST for zero-copy investigation
    var surfaceHandle: UInt64 = 0
    if let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer) {
      surfaceHandle = UInt64(IOSurfaceGetID(ioSurface.takeUnretainedValue()))
      // Log once for zero-copy follow-up
      if getCurrentFrameHandle() == 0 {
        print("[WebGPUCamera] IOSurface handle available: \(surfaceHandle) (logged for zero-copy follow-up)")
      }
    }

    // Copy pixel data to Rust frame slot, passing the IOSurface handle
    let data = Data(bytes: baseAddress, count: dataSize)
    deliverFrame(pixels: data, handle: surfaceHandle)

    // Feed frames to recorder when active
    module?.appendFrameToRecorder(pixels: data, width: Int(width), height: Int(height))
  }
}
