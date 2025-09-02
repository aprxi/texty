//! Utility functions for the text classification library

/// Set panic hook for better error reporting in WASM
#[cfg(feature = "wasm")]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(not(feature = "wasm"))]
pub fn set_panic_hook() {
    // No-op for non-WASM builds
}

/// Log a message (works in both native and WASM environments)
#[cfg(feature = "wasm")]
pub fn log(message: &str) {
    crate::console_log!("{}", message);
}

#[cfg(not(feature = "wasm"))]
pub fn log(message: &str) {
    println!("{}", message);
}

/// Performance timer for measuring execution time (WASM stub)
pub struct Timer {
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        0.0 // Timing not available in WASM stub
    }
    
    pub fn log_elapsed(&self) {
        log(&format!("{}: timing not available in WASM", self.name));
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // No-op for WASM compatibility
    }
}