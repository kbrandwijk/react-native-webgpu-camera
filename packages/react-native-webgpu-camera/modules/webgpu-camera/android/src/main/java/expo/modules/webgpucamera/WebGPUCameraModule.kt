package expo.modules.webgpucamera

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class WebGPUCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("WebGPUCamera")

    // Android implementation pending — iOS-only for now
  }
}
