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

  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, width: Int, height: Int, fps: Int) in
      self.startCapture(deviceId: deviceId, width: width, height: height, fps: fps)
    }

    Function("stopCameraPreview") {
      self.stopCapture()
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
        sync: sync
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

  private func startCapture(deviceId: String, width: Int, height: Int, fps: Int) {
    sessionQueue.async {
      let session = AVCaptureSession()
      session.sessionPreset = .inputPriority

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
