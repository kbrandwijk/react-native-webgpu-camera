//! Android camera — capture will be handled in Kotlin Expo module (Task 3).

pub fn start_preview(_device_id: &str, _width: u32, _height: u32) {
    println!("[webgpu-camera/android] Camera managed by Kotlin Expo module");
}

pub fn stop_preview() {
    println!("[webgpu-camera/android] Camera stopped by Kotlin Expo module");
}
