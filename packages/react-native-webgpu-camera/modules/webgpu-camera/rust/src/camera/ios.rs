//! iOS camera — capture will be handled in Swift Expo module (Task 2).
//! This stub remains for the lib.rs conditional compile.

pub fn start_preview(_device_id: &str, _width: u32, _height: u32) {
    println!("[webgpu-camera/ios] Camera managed by Swift Expo module");
}

pub fn stop_preview() {
    println!("[webgpu-camera/ios] Camera stopped by Swift Expo module");
}
