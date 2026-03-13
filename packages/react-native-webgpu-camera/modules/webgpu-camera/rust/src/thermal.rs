//! Thermal state placeholder.
//! The real implementation lives in the Expo module's native code
//! (Swift/Kotlin) which has direct access to ProcessInfo / PowerManager.

pub fn ios_thermal_state() -> String {
    "nominal".to_string()
}

pub fn android_thermal_state() -> String {
    "nominal".to_string()
}
