//! A prerecorded capture device: plays a WAV file into the pipeline in real
//! time, as if someone were speaking it into a microphone.
//!
//! This exercises the *production* capture path (worker polling, stale-audio
//! clearing, resampling, VAD, ASR) with real speech on machines without a
//! microphone, and makes end-to-end runs against the real models repeatable.
//! The file restarts every time the consumer clears the device — which the
//! microphone worker does when push-to-talk starts — so each recording hears
//! the whole utterance, followed by silence (a real microphone never stops).

use std::path::PathBuf;
use std::time::Instant;

use crate::backends::{MicBackend, MicSource};
use crate::protocol::MicInfo;

pub struct WavMicBackend {
    path: PathBuf,
}

impl WavMicBackend {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl MicBackend for WavMicBackend {
    fn open(&mut self, _device: Option<&str>) -> Result<Box<dyn MicSource>, String> {
        let audio = rpg_speech::read_wav(&self.path)
            .map_err(|e| format!("cannot read {}: {e}", self.path.display()))?;
        Ok(Box::new(WavMicSource {
            name: format!("wav:{}", self.path.display()),
            rate: audio.sample_rate.max(1),
            samples: audio.samples,
            started: Instant::now(),
            delivered: 0,
        }))
    }

    fn list_devices(&mut self) -> Vec<String> {
        vec![format!("wav:{}", self.path.display())]
    }
}

struct WavMicSource {
    name: String,
    rate: u32,
    samples: Vec<f32>,
    started: Instant,
    /// Samples delivered since `started` (file, then silence).
    delivered: usize,
}

impl MicSource for WavMicSource {
    fn info(&self) -> MicInfo {
        MicInfo {
            name: self.name.clone(),
            sample_rate: self.rate,
            channels: 1,
            sample_format: "WAV/PCM16".into(),
        }
    }

    fn try_recv(&mut self) -> Option<Vec<f32>> {
        let due = (self.started.elapsed().as_secs_f64() * self.rate as f64) as usize;
        if due <= self.delivered {
            return None;
        }
        // 10 ms device buffers.
        let n = (due - self.delivered).min(self.rate as usize / 100).max(1);
        let chunk = (self.delivered..self.delivered + n)
            .map(|i| self.samples.get(i).copied().unwrap_or(0.0))
            .collect();
        self.delivered += n;
        Some(chunk)
    }

    fn clear(&mut self) -> usize {
        let due = (self.started.elapsed().as_secs_f64() * self.rate as f64) as usize;
        let dropped = due.saturating_sub(self.delivered);
        self.started = Instant::now();
        self.delivered = 0;
        dropped
    }

    fn dropped_chunks(&self) -> u64 {
        0
    }

    fn take_error(&mut self) -> Option<String> {
        None
    }
}
