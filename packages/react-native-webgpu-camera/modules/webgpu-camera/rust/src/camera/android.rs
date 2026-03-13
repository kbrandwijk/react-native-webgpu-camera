//! Android camera implementation using Camera2.
//!
//! Pipeline:
//! Camera2 -> ImageReader -> Image.getHardwareBuffer()
//!   -> HardwareBuffer handle
//!   -> Store handle in CURRENT_FRAME_HANDLE atomic

use crate::{CURRENT_FRAME_HANDLE, CURRENT_FRAME_PIXELS};
use std::sync::atomic::Ordering;

pub fn start_preview(device_id: &str, width: u32, height: u32) {
    // TODO: Implement Camera2 setup via JNI
    println!(
        "[webgpu-camera/android] start_preview({}, {}x{}) — stub",
        device_id, width, height
    );
}

pub fn stop_preview() {
    CURRENT_FRAME_HANDLE.store(0, Ordering::Relaxed);
    CURRENT_FRAME_PIXELS.lock().unwrap().clear();
    println!("[webgpu-camera/android] stop_preview — stub");
}
