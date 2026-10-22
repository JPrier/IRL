//! Lock-free counters shared between the workers and the simulation.
//!
//! Workers *write* these with relaxed atomic stores; the simulation *reads*
//! them when it builds a snapshot. Nothing here allocates or locks, so a
//! worker (or the capture path) can update telemetry on every chunk without
//! ever contending with the simulation thread.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// An `f32` stored in an `AtomicU32`.
#[derive(Default, Debug)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// Sentinel for "no request" in [`Telemetry::active_inference`].
pub const NONE: u64 = 0;

#[derive(Default, Debug)]
pub struct Telemetry {
    // --- simulation -------------------------------------------------------
    pub ticks: AtomicU64,
    pub ticks_per_second: AtomicF32,
    pub tick_delay_ms: AtomicF32,
    pub max_tick_lag_ms: AtomicF32,
    pub skipped_ticks: AtomicU64,
    /// Snapshots dropped because the renderer was not reading them.
    pub dropped_snapshots: AtomicU64,

    // --- microphone -------------------------------------------------------
    /// Mono samples delivered by the device since start.
    pub mic_samples: AtomicU64,
    pub mic_samples_per_second: AtomicF32,
    pub mic_rms: AtomicF32,
    pub mic_peak: AtomicF32,
    pub mic_dropped_chunks: AtomicU64,
    pub mic_listening: AtomicBool,
    pub vad_speaking: AtomicBool,
    pub recording_ms: AtomicU64,

    // --- workers ----------------------------------------------------------
    pub asr_queue: AtomicUsize,
    pub asr_busy: AtomicBool,
    pub infer_queue: AtomicUsize,
    /// Request id currently inside the model, or [`NONE`].
    pub active_inference: AtomicU64,
    pub tts_queue: AtomicUsize,
    pub tts_synthesizing: AtomicBool,
    pub playback_active: AtomicBool,
    pub playback_underruns: AtomicU64,
    pub playback_buffered_ms: AtomicF32,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Short description of the playback pipeline's current state.
    pub fn playback_status(&self) -> String {
        let synth = self.tts_synthesizing.load(Ordering::Relaxed);
        let playing = self.playback_active.load(Ordering::Relaxed);
        let buffered = self.playback_buffered_ms.get();
        match (synth, playing) {
            (true, true) => format!("synthesizing + playing ({buffered:.0} ms buffered)"),
            (true, false) => "synthesizing".into(),
            (false, true) => format!("playing ({buffered:.0} ms buffered)"),
            (false, false) => "idle".into(),
        }
    }
}
