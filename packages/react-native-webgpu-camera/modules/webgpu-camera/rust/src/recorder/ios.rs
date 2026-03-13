//! iOS video recording using AVAssetWriter.

pub fn start_recorder(output_path: &str, width: u32, height: u32) -> u64 {
    // TODO: Implement AVAssetWriter setup
    println!(
        "[webgpu-camera/ios] start_recorder({}, {}x{}) — stub",
        output_path, width, height
    );
    0
}

pub fn stop_recorder() -> String {
    println!("[webgpu-camera/ios] stop_recorder — stub");
    String::new()
}
