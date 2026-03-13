use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

uniffi::setup_scaffolding!();

// Thread-safe slot for latest camera frame handle.
// Camera callback updates this; JS polls it on render tick.
static CURRENT_FRAME_HANDLE: AtomicU64 = AtomicU64::new(0);
static CURRENT_FRAME_PIXELS: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static FRAME_DIMS: Mutex<FrameDimensions> = Mutex::new(FrameDimensions {
    width: 0,
    height: 0,
    bytes_per_row: 0,
});

#[derive(uniffi::Record, Clone)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

#[uniffi::export]
pub fn start_camera_preview(device_id: String, width: u32, height: u32) {
    println!(
        "[webgpu-camera] start_camera_preview({}, {}x{})",
        device_id, width, height
    );

    let mut dims = FRAME_DIMS.lock().unwrap();
    dims.width = width;
    dims.height = height;
    dims.bytes_per_row = width * 4;

    #[cfg(target_os = "ios")]
    camera::ios::start_preview(&device_id, width, height);

    #[cfg(target_os = "android")]
    camera::android::start_preview(&device_id, width, height);
}

#[uniffi::export]
pub fn stop_camera_preview() {
    #[cfg(target_os = "ios")]
    camera::ios::stop_preview();

    #[cfg(target_os = "android")]
    camera::android::stop_preview();
}

#[uniffi::export]
pub fn get_current_frame_handle() -> u64 {
    CURRENT_FRAME_HANDLE.load(Ordering::Relaxed)
}

#[uniffi::export]
pub fn get_current_frame_pixels() -> Vec<u8> {
    CURRENT_FRAME_PIXELS.lock().unwrap().clone()
}

#[uniffi::export]
pub fn get_frame_dimensions() -> FrameDimensions {
    FRAME_DIMS.lock().unwrap().clone()
}

#[uniffi::export]
pub fn start_test_recorder(output_path: String, width: u32, height: u32) -> u64 {
    #[cfg(target_os = "ios")]
    return recorder::ios::start_recorder(&output_path, width, height);

    #[cfg(target_os = "android")]
    return recorder::android::start_recorder(&output_path, width, height);

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = (output_path, width, height);
        0
    }
}

#[uniffi::export]
pub fn stop_test_recorder() -> String {
    #[cfg(target_os = "ios")]
    return recorder::ios::stop_recorder();

    #[cfg(target_os = "android")]
    return recorder::android::stop_recorder();

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    String::new()
}

#[uniffi::export]
pub fn get_thermal_state() -> String {
    #[cfg(target_os = "ios")]
    return thermal::ios_thermal_state();

    #[cfg(target_os = "android")]
    return thermal::android_thermal_state();

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    "nominal".to_string()
}

mod camera;
mod recorder;
mod thermal;
