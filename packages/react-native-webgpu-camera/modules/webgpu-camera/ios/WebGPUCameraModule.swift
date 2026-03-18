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
  var isAppleLog = false
  var isHDR = false  // true for appleLog OR hlgBT2020 — both deliver 10-bit YUV
  var isYUV422 = false  // true when camera delivers 4:2:2 (x422) instead of 4:2:0 (x420)
  var useDepth = false
  var isLiDARYUV = false  // true when LiDAR device delivers 8-bit YUV (420f) instead of BGRA
  private var depthOutput: AVCaptureDepthDataOutput?
  private var depthDelegate: DepthDelegate?

  // Saved start parameters for session restart (e.g. when depth is added after start)
  private var lastDeviceId: String?
  private var lastNativeHandle: Int?
  private var lastColorSpace: String?

  /// Stored AVCaptureDevice.Format arrays, keyed by device position.
  /// Rebuilt on each getFormats() call. nativeHandle indexes into these.
  var storedBackFormats: [AVCaptureDevice.Format] = []
  var storedFrontFormats: [AVCaptureDevice.Format] = []

  var activeWidth: Int = 0
  var activeHeight: Int = 0
  var activeFps: Int = 0

  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, nativeHandle: Int, colorSpace: String, useDepth: Bool) in
      self.useDepth = useDepth
      self.startCapture(deviceId: deviceId, nativeHandle: nativeHandle, colorSpace: colorSpace)
    }

    Function("stopCameraPreview") {
      self.stopCapture()
    }

    Function("getFormats") { (deviceId: String) -> [[String: Any]] in
        let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back

        // Try LiDAR depth camera first (has both video + depth formats),
        // fall back to wide angle
        var camera: AVCaptureDevice?
        if #available(iOS 15.4, *) {
          camera = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: position)
        }
        if camera == nil {
          camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
        }
        guard let camera = camera else {
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
                "supportsDepth": !format.supportedDepthDataFormats.isEmpty,
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
      let appleLog = config["appleLog"] as? Bool ?? false
      let resourcesRaw = config["resources"] as? [[String: Any]] ?? []
      let passInputsRaw = config["passInputs"] as? [[String: Any]] ?? []
      let textureOutputPasses = config["textureOutputPasses"] as? [NSNumber] ?? []
      let useDepth = config["useDepth"] as? Bool ?? false
      self.useDepth = useDepth

      let bufferSpecs = bufferSpecsRaw.map { spec in
        spec.map { $0 }
      }

      self.isAppleLog = appleLog

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
        appleLog: appleLog,
        useDepth: useDepth,
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
        print("[WebGPUCamera] Multi-pass pipeline setup OK: \(shaders.count) passes, \(width)x\(height), useDepth=\(useDepth)")

        // If depth was requested but capture session started without it,
        // Depth output is now configured in startCapture via useCamera({ useDepth: true })
        // No mid-session depth addition needed.
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
    // Save for potential session restart (e.g. when depth is added later)
    self.lastDeviceId = deviceId
    self.lastNativeHandle = nativeHandle
    self.lastColorSpace = colorSpace

    NSLog("[WebGPUCamera] startCapture ENTER: device=\(deviceId) nativeHandle=\(nativeHandle) colorSpace=\(colorSpace) useDepth=\(self.useDepth)")
    sessionQueue.async { [weak self] in
      guard let self = self else {
        NSLog("[WebGPUCamera] startCapture: self is nil, aborting")
        return
      }
      NSLog("[WebGPUCamera] startCapture: step 1 — creating session")
      let session = AVCaptureSession()
      session.sessionPreset = .inputPriority
      session.automaticallyConfiguresCaptureDeviceForWideColor = false

      let position: AVCaptureDevice.Position = deviceId == "front" ? .front : .back
      NSLog("[WebGPUCamera] startCapture: step 2 — getting camera device (position=\(position.rawValue))")

      // Prefer LiDAR depth camera (supports both video + depth, same formats).
      // Fall back to wide angle if LiDAR not available.
      var camera: AVCaptureDevice?
      if #available(iOS 15.4, *) {
        camera = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: position)
      }
      if camera == nil {
        camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
        if self.useDepth {
          NSLog("[WebGPUCamera] WARNING: LiDAR not available, depth will not work")
          self.useDepth = false
        }
      }
      guard let camera = camera else {
        NSLog("[WebGPUCamera] No camera found for position: \(deviceId)")
        return
      }
      NSLog("[WebGPUCamera] startCapture: step 3 — creating input")

      guard let input = try? AVCaptureDeviceInput(device: camera) else {
        NSLog("[WebGPUCamera] Could not create camera input")
        return
      }
      NSLog("[WebGPUCamera] startCapture: step 4 — adding input to session")
      if session.canAddInput(input) {
        session.addInput(input)
      }

      // Format selection
      NSLog("[WebGPUCamera] startCapture: step 5 — locking camera for configuration")
      do {
        try camera.lockForConfiguration()
        NSLog("[WebGPUCamera] startCapture: step 5a — camera locked OK")

        if nativeHandle >= 0 {
          // User selected a specific format
          let storedFormats = position == .back ? self.storedBackFormats : self.storedFrontFormats
          NSLog("[WebGPUCamera] startCapture: step 5b — storedFormats count=\(storedFormats.count), nativeHandle=\(nativeHandle)")
          if nativeHandle < storedFormats.count {
            let selectedFormat = storedFormats[nativeHandle]
            NSLog("[WebGPUCamera] startCapture: step 5c — setting activeFormat")
            camera.activeFormat = selectedFormat

            // Set FPS to format's max
            var bestMaxFps: Float64 = 30
            for range in selectedFormat.videoSupportedFrameRateRanges {
              bestMaxFps = max(bestMaxFps, range.maxFrameRate)
            }
            NSLog("[WebGPUCamera] startCapture: step 5d — setting FPS to \(bestMaxFps)")
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
        NSLog("[WebGPUCamera] startCapture: step 6 — setting color space '\(colorSpace)'")
        let targetColorSpace = self.mapColorSpace(colorSpace)
        let supportedSpaces = camera.activeFormat.supportedColorSpaces.map { "\($0.rawValue)" }.joined(separator: ",")
        NSLog("[WebGPUCamera] startCapture: step 6a — supported color spaces: [\(supportedSpaces)], target=\(targetColorSpace.rawValue)")
        if camera.activeFormat.supportedColorSpaces.contains(targetColorSpace) {
          NSLog("[WebGPUCamera] startCapture: step 6b — setting activeColorSpace")
          camera.activeColorSpace = targetColorSpace
          // Only set HDR flags if the format actually supports the requested color space
          self.isAppleLog = (colorSpace == "appleLog")
          self.isHDR = (colorSpace == "appleLog" || colorSpace == "hlgBT2020")
          NSLog("[WebGPUCamera] startCapture: step 6c — isAppleLog=\(self.isAppleLog) isHDR=\(self.isHDR)")
        } else {
          NSLog("[WebGPUCamera] Color space '\(colorSpace)' not supported by active format, falling back to sRGB")
          camera.activeColorSpace = .sRGB
          self.isAppleLog = false
          self.isHDR = false
        }

        NSLog("[WebGPUCamera] startCapture: step 7 — unlocking camera")
        camera.unlockForConfiguration()
      } catch {
        NSLog("[WebGPUCamera] Failed to lock camera for configuration: \(error)")
      }

      // Frame output setup
      NSLog("[WebGPUCamera] startCapture: step 8 — creating video output (isHDR=\(self.isHDR))")
      let dims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
      let width = Int(dims.width)
      let height = Int(dims.height)

      let output = AVCaptureVideoDataOutput()
      output.alwaysDiscardsLateVideoFrames = true

      // Always create FrameDelegate for video frames.
      // Depth is handled separately via DepthDelegate (added later by setupMultiPassPipeline).
      NSLog("[WebGPUCamera] startCapture: step 9 — setting up frame delegate")
      let delegate = FrameDelegate(width: UInt32(width), height: UInt32(height), module: self)
      self.frameDelegate = delegate
      output.setSampleBufferDelegate(delegate, queue: self.frameQueue)

      NSLog("[WebGPUCamera] startCapture: step 10 — adding output to session")
      if session.canAddOutput(output) {
        session.addOutput(output)
      }

      // Depth output — added here in the same session setup, not later
      if self.useDepth {
        NSLog("[WebGPUCamera] startCapture: step 10-depth — adding depth output")
        let depthOut = AVCaptureDepthDataOutput()
        depthOut.isFilteringEnabled = true
        depthOut.alwaysDiscardsLateDepthData = true
        if session.canAddOutput(depthOut) {
          session.addOutput(depthOut)
          self.depthOutput = depthOut

          let depthDel = DepthDelegate()
          depthOut.setDelegate(depthDel, callbackQueue: self.frameQueue)
          self.depthDelegate = depthDel
          delegate.depthDelegate = depthDel

          // Set activeDepthDataFormat to DepthFloat16
          do {
            try camera.lockForConfiguration()
            if let depthFormat = camera.activeFormat.supportedDepthDataFormats.first(where: {
              CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
            }) {
              camera.activeDepthDataFormat = depthFormat
              NSLog("[WebGPUCamera] startCapture: activeDepthDataFormat set to DepthFloat16")
            } else {
              NSLog("[WebGPUCamera] startCapture: WARNING — no DepthFloat16 format available")
            }
            camera.unlockForConfiguration()
          } catch {
            NSLog("[WebGPUCamera] startCapture: failed to lock camera for depth format: \(error)")
          }

          NSLog("[WebGPUCamera] startCapture: depth output + delegate configured")
        } else {
          NSLog("[WebGPUCamera] startCapture: WARNING — could not add depth output")
          self.useDepth = false
        }
      }

      // Query available pixel formats AFTER adding output to session
      // (available formats depend on session configuration and active format)
      let available = output.availableVideoPixelFormatTypes  // [OSType]
      let availableHex = available.map { String(format: "0x%08x", $0) }
      NSLog("[WebGPUCamera] startCapture: step 10a — availableVideoPixelFormatTypes: \(availableHex)")
      if self.isHDR {
        // Prefer 4:2:2 (x422) — matches camera native format, Blackmagic uses this.
        // Must disable automaticallyConfiguresCaptureDeviceForWideColor to preserve Apple Log.
        let vr422 = OSType(kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange)
        let vr420 = OSType(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)

        let chosen: OSType
        if available.contains(vr422) {
          chosen = vr422
          self.isYUV422 = true
          NSLog("[WebGPUCamera] startCapture: step 10b — using 10-bit YUV 4:2:2 VideoRange (x422)")
        } else if available.contains(vr420) {
          chosen = vr420
          self.isYUV422 = false
          NSLog("[WebGPUCamera] startCapture: step 10b — using 10-bit YUV 4:2:0 VideoRange (x420)")
        } else {
          NSLog("[WebGPUCamera] startCapture: step 10b — NO 10-bit YUV format available, falling back to BGRA")
          chosen = OSType(kCVPixelFormatType_32BGRA)
          self.isHDR = false
          self.isAppleLog = false
          self.isYUV422 = false
        }
        output.videoSettings = [
          kCVPixelBufferPixelFormatTypeKey as String: chosen
        ]
      } else if self.useDepth {
        // LiDAR depth camera — try video range YUV first (420v matches Dawn's NV12),
        // then full range (420f), then BGRA as fallback
        let vr420 = OSType(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        let fr420 = OSType(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        if available.contains(vr420) {
          output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: vr420
          ]
          self.isLiDARYUV = true
          NSLog("[WebGPUCamera] startCapture: step 10b — LiDAR: using 8-bit YUV 4:2:0 VideoRange (420v)")
        } else if available.contains(fr420) {
          output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: fr420
          ]
          self.isLiDARYUV = true
          NSLog("[WebGPUCamera] startCapture: step 10b — LiDAR: using 8-bit YUV 4:2:0 FullRange (420f)")
        } else {
          NSLog("[WebGPUCamera] startCapture: step 10b — LiDAR: no YUV available, trying BGRA")
          output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
          ]
          self.isLiDARYUV = false
        }
      } else {
        NSLog("[WebGPUCamera] startCapture: step 10b — requesting BGRA pixel format")
        output.videoSettings = [
          kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        self.isLiDARYUV = false
      }

      self.activeWidth = width
      self.activeHeight = height

      NSLog("[WebGPUCamera] startCapture: step 11 — calling session.startRunning()")
      session.startRunning()
      NSLog("[WebGPUCamera] startCapture: step 12 — session started OK")

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
      self.depthDelegate = nil
      self.depthOutput = nil
      print("[WebGPUCamera] Camera stopped")
    }
  }

}

private class FrameDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  let width: UInt32
  let height: UInt32
  weak var module: WebGPUCameraModule?
  var depthDelegate: DepthDelegate?

  init(width: UInt32, height: UInt32, module: WebGPUCameraModule) {
    self.width = width
    self.height = height
    self.module = module
  }

  private var frameCount: Int = 0

  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    frameCount += 1
    if frameCount <= 3 || frameCount % 300 == 0 {
      let hasBridge = module?.dawnBridge != nil
      let hasDepth = depthDelegate?.latestDepth != nil
      NSLog("[FrameDelegate] frame #%d, module=%d, dawnBridge=%d, depth=%d", frameCount, module != nil ? 1 : 0, hasBridge ? 1 : 0, hasDepth ? 1 : 0)
    }
    if frameCount == 1 {
      let fd = CMSampleBufferGetFormatDescription(sampleBuffer)
      if let fd = fd {
        let exts = CMFormatDescriptionGetExtensions(fd) as? [String: Any]
        let tf = exts?[kCMFormatDescriptionExtension_TransferFunction as String]
        let matrix = exts?[kCMFormatDescriptionExtension_YCbCrMatrix as String]
        let primaries = exts?[kCMFormatDescriptionExtension_ColorPrimaries as String]
        NSLog("[FrameDelegate] CMFormatDesc transfer=%@ matrix=%@ primaries=%@",
              tf as? String ?? "(none)", matrix as? String ?? "(none)", primaries as? String ?? "(none)")
      }
    }

    // Grab latest depth buffer (may be nil or from a slightly different timestamp — acceptable)
    let depthBuffer = depthDelegate?.latestDepth

    // Run Dawn compute pipeline with video + optional depth
    module?.dawnBridge?.processFrame(pixelBuffer, depthBuffer: depthBuffer)
  }
}

/// Caches the latest depth data from AVCaptureDepthDataOutput.
/// Video FrameDelegate grabs the latest on each video frame.
private class DepthDelegate: NSObject, AVCaptureDepthDataOutputDelegate {
  var latestDepth: CVPixelBuffer?
  private var frameCount: Int = 0

  func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData, timestamp: CMTime, connection: AVCaptureConnection) {
    latestDepth = depthData.depthDataMap
    frameCount += 1
    if frameCount <= 3 {
      let fmt = CVPixelBufferGetPixelFormatType(depthData.depthDataMap)
      let w = CVPixelBufferGetWidth(depthData.depthDataMap)
      let h = CVPixelBufferGetHeight(depthData.depthDataMap)
      NSLog("[DepthDelegate] frame #%d, fmt=0x%08x, %dx%d", frameCount, fmt, w, h)
    }
  }
}
