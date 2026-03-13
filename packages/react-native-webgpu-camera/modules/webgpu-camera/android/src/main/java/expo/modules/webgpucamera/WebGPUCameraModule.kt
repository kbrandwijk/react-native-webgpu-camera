package expo.modules.webgpucamera

import android.os.PowerManager
import android.content.Context
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class WebGPUCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { deviceId: String, width: Int, height: Int ->
      // TODO: Call Rust start_camera_preview via UniFFI bindings
      println("[WebGPUCamera] startCameraPreview($deviceId, ${width}x${height})")
    }

    Function("stopCameraPreview") {
      // TODO: Call Rust stop_camera_preview
      println("[WebGPUCamera] stopCameraPreview")
    }

    Function("getCurrentFrameHandle") {
      // TODO: Call Rust get_current_frame_handle
      0L
    }

    Function("getCurrentFramePixels") {
      // TODO: Call Rust get_current_frame_pixels
      ByteArray(0)
    }

    Function("getFrameDimensions") {
      // TODO: Call Rust get_frame_dimensions
      mapOf("width" to 0, "height" to 0, "bytesPerRow" to 0)
    }

    Function("startTestRecorder") { outputPath: String, width: Int, height: Int ->
      // TODO: Call Rust start_test_recorder
      0L
    }

    Function("stopTestRecorder") {
      // TODO: Call Rust stop_test_recorder
      ""
    }

    Function("getThermalState") {
      val context = appContext.reactContext ?: return@Function "nominal"
      val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
      when (pm?.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "nominal"
        PowerManager.THERMAL_STATUS_LIGHT -> "fair"
        PowerManager.THERMAL_STATUS_MODERATE -> "fair"
        PowerManager.THERMAL_STATUS_SEVERE -> "serious"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "critical"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "critical"
        else -> "nominal"
      }
    }
  }
}
