package expo.modules.webgpucamera

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.nio.ByteBuffer

class WebGPUCameraModule : Module() {
  private var cameraDevice: CameraDevice? = null
  private var captureSession: CameraCaptureSession? = null
  private var imageReader: ImageReader? = null
  private var backgroundThread: HandlerThread? = null
  private var backgroundHandler: Handler? = null

  override fun definition() = ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { deviceId: String, width: Int, height: Int ->
      startCapture(deviceId, width, height)
    }

    Function("stopCameraPreview") {
      stopCapture()
    }

    Function("getCurrentFrameHandle") {
      uniffi.webgpu_camera.getCurrentFrameHandle().toLong()
    }

    Function("getCurrentFramePixels") {
      uniffi.webgpu_camera.getCurrentFramePixels().toByteArray()
    }

    Function("getFrameDimensions") {
      val dims = uniffi.webgpu_camera.getFrameDimensions()
      mapOf("width" to dims.width.toInt(), "height" to dims.height.toInt(), "bytesPerRow" to dims.bytesPerRow.toInt())
    }

    Function("getFrameCounter") {
      uniffi.webgpu_camera.getFrameCounter().toLong()
    }

    Function("startTestRecorder") { outputPath: String, width: Int, height: Int ->
      uniffi.webgpu_camera.startTestRecorder(outputPath, width.toUInt(), height.toUInt()).toLong()
    }

    Function("stopTestRecorder") {
      uniffi.webgpu_camera.stopTestRecorder()
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

  private fun startCapture(deviceId: String, width: Int, height: Int) {
    val context = appContext.reactContext ?: return

    // Start background thread
    backgroundThread = HandlerThread("WebGPUCamera").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)

    // Set frame dimensions in Rust
    uniffi.webgpu_camera.setFrameDimensions(width.toUInt(), height.toUInt(), (width * 4).toUInt())

    // Create ImageReader for YUV output
    imageReader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 2).apply {
      setOnImageAvailableListener({ reader ->
        val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
        try {
          // Convert YUV to BGRA and deliver to Rust frame slot
          val yPlane = image.planes[0]
          val uPlane = image.planes[1]
          val vPlane = image.planes[2]

          // WARNING: This per-pixel loop (~50-100ms at 1080p) will NOT sustain 30fps.
          // Spike will run at reduced framerate on Android. Acceptable for validation.
          val bgra = yuvToBgra(
            yPlane.buffer, uPlane.buffer, vPlane.buffer,
            width, height,
            yPlane.rowStride, uPlane.rowStride, uPlane.pixelStride
          )

          uniffi.webgpu_camera.deliverFrame(bgra.asUByteArray().toList(), 0u)
        } finally {
          image.close()
        }
      }, backgroundHandler)
    }

    // Open camera
    val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    val facing = if (deviceId == "front") CameraCharacteristics.LENS_FACING_FRONT
                 else CameraCharacteristics.LENS_FACING_BACK

    val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
      val chars = cameraManager.getCameraCharacteristics(id)
      chars.get(CameraCharacteristics.LENS_FACING) == facing
    } ?: return

    try {
      cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
          cameraDevice = camera
          createCaptureSession(camera, width, height)
        }

        override fun onDisconnected(camera: CameraDevice) {
          camera.close()
          cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
          camera.close()
          cameraDevice = null
          println("[WebGPUCamera] Camera error: $error")
        }
      }, backgroundHandler)
    } catch (e: SecurityException) {
      println("[WebGPUCamera] Camera permission not granted")
    }
  }

  private fun createCaptureSession(camera: CameraDevice, width: Int, height: Int) {
    val surface = imageReader?.surface ?: return

    camera.createCaptureSession(
      listOf(surface),
      object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) {
          captureSession = session
          val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(surface)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
          }
          session.setRepeatingRequest(request.build(), null, backgroundHandler)
          println("[WebGPUCamera] Camera capture session started")
        }

        override fun onConfigureFailed(session: CameraCaptureSession) {
          println("[WebGPUCamera] Capture session configuration failed")
        }
      },
      backgroundHandler
    )
  }

  private fun stopCapture() {
    captureSession?.close()
    captureSession = null
    cameraDevice?.close()
    cameraDevice = null
    imageReader?.close()
    imageReader = null
    backgroundThread?.quitSafely()
    backgroundThread = null
    backgroundHandler = null
    println("[WebGPUCamera] Camera stopped")
  }

  private fun yuvToBgra(
    yBuf: ByteBuffer, uBuf: ByteBuffer, vBuf: ByteBuffer,
    width: Int, height: Int,
    yRowStride: Int, uvRowStride: Int, uvPixelStride: Int
  ): ByteArray {
    val bgra = ByteArray(width * height * 4)
    for (row in 0 until height) {
      for (col in 0 until width) {
        val yIdx = row * yRowStride + col
        val uvRow = row / 2
        val uvCol = col / 2
        val uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride

        val y = (yBuf.get(yIdx).toInt() and 0xFF).toFloat()
        val u = (uBuf.get(uvIdx).toInt() and 0xFF).toFloat() - 128f
        val v = (vBuf.get(uvIdx).toInt() and 0xFF).toFloat() - 128f

        val r = (y + 1.370705f * v).toInt().coerceIn(0, 255)
        val g = (y - 0.337633f * u - 0.698001f * v).toInt().coerceIn(0, 255)
        val b = (y + 1.732446f * u).toInt().coerceIn(0, 255)

        val outIdx = (row * width + col) * 4
        bgra[outIdx] = b.toByte()
        bgra[outIdx + 1] = g.toByte()
        bgra[outIdx + 2] = r.toByte()
        bgra[outIdx + 3] = 0xFF.toByte()
      }
    }
    return bgra
  }
}
