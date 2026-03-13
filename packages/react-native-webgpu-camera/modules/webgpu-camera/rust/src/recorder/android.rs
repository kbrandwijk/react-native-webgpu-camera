//! Android video recording using MediaRecorder.

pub fn start_recorder(output_path: &str, width: u32, height: u32) -> u64 {
    // TODO: Implement MediaRecorder setup via JNI
    println!(
        "[webgpu-camera/android] start_recorder({}, {}x{}) — stub",
        output_path, width, height
    );
    0
}

pub fn stop_recorder() -> String {
    println!("[webgpu-camera/android] stop_recorder — stub");
    String::new()
}
