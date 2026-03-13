//! iOS camera implementation using AVCaptureSession.
//!
//! Pipeline:
//! AVCaptureSession -> AVCaptureVideoDataOutput
//!   -> CMSampleBufferGetImageBuffer() -> CVPixelBuffer
//!   -> CVPixelBufferGetIOSurface() -> IOSurface handle
//!   -> Store handle in CURRENT_FRAME_HANDLE atomic

use crate::{CURRENT_FRAME_HANDLE, CURRENT_FRAME_PIXELS};
use std::sync::atomic::Ordering;

pub fn start_preview(device_id: &str, width: u32, height: u32) {
    // TODO: Implement AVCaptureSession setup
    println!(
        "[webgpu-camera/ios] start_preview({}, {}x{}) — stub",
        device_id, width, height
    );
}

pub fn stop_preview() {
    CURRENT_FRAME_HANDLE.store(0, Ordering::Relaxed);
    CURRENT_FRAME_PIXELS.lock().unwrap().clear();
    println!("[webgpu-camera/ios] stop_preview — stub");
}
