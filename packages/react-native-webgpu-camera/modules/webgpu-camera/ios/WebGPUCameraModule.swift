import ExpoModulesCore

public class WebGPUCameraModule: Module {
  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, width: Int, height: Int) in
      // TODO: Call Rust start_camera_preview via UniFFI bindings
      print("[WebGPUCamera] startCameraPreview(\(deviceId), \(width)x\(height))")
    }

    Function("stopCameraPreview") {
      // TODO: Call Rust stop_camera_preview
      print("[WebGPUCamera] stopCameraPreview")
    }

    Function("getCurrentFrameHandle") { () -> Int in
      // TODO: Call Rust get_current_frame_handle
      return 0
    }

    Function("getCurrentFramePixels") { () -> Data in
      // TODO: Call Rust get_current_frame_pixels
      return Data()
    }

    Function("getFrameDimensions") { () -> [String: Any] in
      // TODO: Call Rust get_frame_dimensions
      return ["width": 0, "height": 0, "bytesPerRow": 0]
    }

    Function("startTestRecorder") { (outputPath: String, width: Int, height: Int) -> Int in
      // TODO: Call Rust start_test_recorder
      return 0
    }

    Function("stopTestRecorder") { () -> String in
      // TODO: Call Rust stop_test_recorder
      return ""
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
  }
}
