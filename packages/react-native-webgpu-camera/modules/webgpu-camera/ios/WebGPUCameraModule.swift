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

  /// Stored AVCaptureDevice.Format arrays, keyed by device position.
  /// Rebuilt on each getFormats() call. nativeHandle indexes into these.
  var storedBackFormats: [AVCaptureDevice.Format] = []
  var storedFrontFormats: [AVCaptureDevice.Format] = []

  var activeWidth: Int = 0
  var activeHeight: Int = 0
  var activeFps: Int = 0

  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, nativeHandle: Int, colorSpace: String) in
      self.startCapture(deviceId: deviceId, nativeHandle: nativeHandle, colorSpace: colorSpace)
    }

    Function("stopCameraPreview") {
      self.stopCapture()
    }

    Function("getFormats") { (deviceId: String) -> [[String: Any]] in
        let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back

        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: position
        ) else {
            return []
        }

        let formats = camera.formats
        // Store for later lookup by nativeHandle
        if position == .back {
            self.storedBackFormats = formats
        } else {
            self.storedFrontFormats = formats
        }

        return formats.enumerated().map { (index, format) in
            let desc = format.formatDescription
            let dims = CMVideoFormatDescriptionGetDimensions(desc)

            // FPS: min of all minFrameRates, max of all maxFrameRates
            var minFps: Float64 = .greatestFiniteMagnitude
            var maxFps: Float64 = 0
            for range in format.videoSupportedFrameRateRanges {
                minFps = min(minFps, range.minFrameRate)
                maxFps = max(maxFps, range.maxFrameRate)
            }
            if minFps == .greatestFiniteMagnitude { minFps = 0 }

            // Color spaces
            let colorSpaces: [String] = format.supportedColorSpaces.map { cs in
                switch cs {
                case .sRGB: return "sRGB"
                case .P3_D65: return "p3D65"
                case .HLG_BT2020: return "hlgBT2020"
                @unknown default:
                    if #available(iOS 17.0, *), cs == .appleLog { return "appleLog" }
                    return "unknown"
                }
            }

            // Stabilization modes
            var stabilizationModes: [String] = ["off"]
            if format.isVideoStabilizationModeSupported(.standard) {
                stabilizationModes.append("standard")
            }
            if format.isVideoStabilizationModeSupported(.cinematic) {
                stabilizationModes.append("cinematic")
            }
            if format.isVideoStabilizationModeSupported(.cinematicExtended) {
                stabilizationModes.append("cinematicExtended")
            }

            let isHDR = colorSpaces.contains("hlgBT2020")

            return [
                "width": Int(dims.width),
                "height": Int(dims.height),
                "minFps": minFps,
                "maxFps": maxFps,
                "pixelFormat": "bgra",
                "supportedColorSpaces": colorSpaces,
                "isHDR": isHDR,
                "stabilizationModes": stabilizationModes,
                "fieldOfView": format.videoFieldOfView,
                "isBinned": format.isVideoBinned,
                "isMultiCamSupported": format.isMultiCamSupported,
                "nativeHandle": index,
            ] as [String: Any]
        }
    }

    Function("getActiveFormatInfo") { () -> [String: Any]? in
        if self.activeWidth == 0 { return nil }
        return [
            "width": self.activeWidth,
            "height": self.activeHeight,
            "fps": self.activeFps,
        ]
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

    Function("setupMultiPassPipeline") { (config: [String: Any]) -> Bool in
      guard let shaders = config["shaders"] as? [String],
            let width = config["width"] as? Int,
            let height = config["height"] as? Int else {
        print("[WebGPUCamera] setupMultiPassPipeline: missing required config fields")
        return false
      }

      let bufferSpecsRaw = config["buffers"] as? [[NSNumber]] ?? []
      let useCanvas = config["useCanvas"] as? Bool ?? false
      let sync = config["sync"] as? Bool ?? false
      let resourcesRaw = config["resources"] as? [[String: Any]] ?? []
      let passInputsRaw = config["passInputs"] as? [[String: Any]] ?? []
      let textureOutputPasses = (config["textureOutputPasses"] as? [NSNumber] ?? []).map { $0.intValue }

      let bufferSpecs = bufferSpecsRaw.map { spec in
        spec.map { $0 }
      }

      // Clean up any existing bridge before creating a new one
      self.dawnBridge?.cleanup()
      self.dawnBridge = nil

      let bridge = DawnPipelineBridge()
      let ok = bridge.setupMultiPass(
        withShaders: shaders,
        width: Int32(width),
        height: Int32(height),
        bufferSpecs: bufferSpecs,
        useCanvas: useCanvas,
        sync: sync,
        resources: resourcesRaw,
        passInputs: passInputsRaw,
        textureOutputPasses: textureOutputPasses
      )

      if ok {
        self.dawnBridge = bridge
        self.computeSetup = true
        if let runtime = try? self.appContext?.runtime {
          bridge.installJSIBindings(runtime)
        } else {
          print("[WebGPUCamera] WARNING: Could not access runtime for JSI bindings")
        }
        print("[WebGPUCamera] Multi-pass pipeline setup OK: \(shaders.count) passes, \(width)x\(height)")
      } else {
        print("[WebGPUCamera] Multi-pass pipeline setup FAILED")
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

  private func startCapture(deviceId: String, nativeHandle: Int, colorSpace: String) {
    sessionQueue.async { [weak self] in
      guard let self = self else { return }
      let session = AVCaptureSession()
      session.sessionPreset = .inputPriority

      let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back

      guard let camera = AVCaptureDevice.default(
        .builtInWideAngleCamera,
        for: .video,
        position: position
      ) else {
        NSLog("[WebGPUCamera] No camera found for position: \(deviceId)")
        return
      }

      guard let input = try? AVCaptureDeviceInput(device: camera) else {
        NSLog("[WebGPUCamera] Could not create camera input")
        return
      }
      if session.canAddInput(input) {
        session.addInput(input)
      }

      // Format selection
      do {
        try camera.lockForConfiguration()

        if nativeHandle >= 0 {
          // User selected a specific format
          let storedFormats = position == .back ? self.storedBackFormats : self.storedFrontFormats
          if nativeHandle < storedFormats.count {
            let selectedFormat = storedFormats[nativeHandle]
            camera.activeFormat = selectedFormat

            // Set FPS to format's max
            var bestMaxFps: Float64 = 30
            for range in selectedFormat.videoSupportedFrameRateRanges {
              bestMaxFps = max(bestMaxFps, range.maxFrameRate)
            }
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(bestMaxFps))
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(bestMaxFps))
            self.activeFps = Int(bestMaxFps)

            NSLog("[WebGPUCamera] Set format: \(CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription).width)x\(CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription).height) @ \(bestMaxFps)fps")
          } else {
            NSLog("[WebGPUCamera] nativeHandle \(nativeHandle) out of range (\(storedFormats.count) formats), using default")
            self.applyDefaultFormat(camera: camera)
          }
        } else {
          // No format specified — apply default (1080p/30fps)
          self.applyDefaultFormat(camera: camera)
        }

        // Color space — validate against format, then set explicitly (some formats default to P3)
        let targetColorSpace = self.mapColorSpace(colorSpace)
        if camera.activeFormat.supportedColorSpaces.contains(targetColorSpace) {
          camera.activeColorSpace = targetColorSpace
        } else {
          NSLog("[WebGPUCamera] Color space '\(colorSpace)' not supported by active format, falling back to sRGB")
          camera.activeColorSpace = .sRGB
        }

        camera.unlockForConfiguration()
      } catch {
        NSLog("[WebGPUCamera] Failed to lock camera for configuration: \(error)")
      }

      // Frame output setup
      let dims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
      let width = Int(dims.width)
      let height = Int(dims.height)

      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
      output.alwaysDiscardsLateVideoFrames = true

      let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
      self.frameDelegate = delegate
      output.setSampleBufferDelegate(delegate, queue: self.frameQueue)

      if session.canAddOutput(output) {
        session.addOutput(output)
      }

      self.activeWidth = width
      self.activeHeight = height

      session.startRunning()

      self.captureSession = session
      self.dataOutput = output

      NSLog("[WebGPUCamera] Camera started: \(width)x\(height), colorSpace=\(colorSpace)")
    }
  }

  private func mapColorSpace(_ name: String) -> AVCaptureColorSpace {
    switch name {
    case "p3D65": return .P3_D65
    case "hlgBT2020": return .HLG_BT2020
    case "appleLog":
      if #available(iOS 17.0, *) { return .appleLog }
      return .sRGB
    default: return .sRGB
    }
  }

  /// Default format selection: closest to 1080p, maxFps >= 30, prefer non-binned, sRGB-capable
  private func applyDefaultFormat(camera: AVCaptureDevice) {
    var bestFormat: AVCaptureDevice.Format?
    var bestScore = Int.max

    for format in camera.formats {
      let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)

      // Must support sRGB
      let supportsSRGB = format.supportedColorSpaces.contains(.sRGB)
      if !supportsSRGB { continue }

      // Must support >= 30fps
      var maxFps: Float64 = 0
      for range in format.videoSupportedFrameRateRanges {
        maxFps = max(maxFps, range.maxFrameRate)
      }
      if maxFps < 30 { continue }

      // Score: distance from 1080p (lower is better)
      let resScore = abs(Int(dims.width) - 1920) + abs(Int(dims.height) - 1080)
      // Prefer lower maxFps (don't overshoot) — small penalty
      let fpsScore = Int(maxFps - 30)
      // Prefer non-binned
      let binScore = format.isVideoBinned ? 100 : 0

      let score = resScore + fpsScore + binScore
      if score < bestScore {
        bestScore = score
        bestFormat = format
      }
    }

    if let format = bestFormat {
      camera.activeFormat = format

      // Default to 30fps
      camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
      camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
      self.activeFps = 30

      let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
      NSLog("[WebGPUCamera] Default format: \(dims.width)x\(dims.height) @ 30fps")
    } else {
      NSLog("[WebGPUCamera] No suitable default format found, using device default")
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
  }
}
