//! Real audio-device I/O: microphone capture and speaker playback via `cpal`.
//!
//! Design constraints (see the design brief):
//! * audio callbacks are **nonblocking** and never run inference or disk I/O;
//! * buffers are bounded, so a slow consumer drops audio in a controlled way
//!   rather than growing memory without limit;
//! * sample-format conversion happens in the callback (cheap), while heavier
//!   resampling happens on the consumer side.
//!
//! On machines without a microphone or output device, construction fails with
//! [`SpeechError::Device`]; the WAV-fixture paths remain fully testable without
//! any device at all.

use rpg_domain::ports::SpeechError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

/// Description of a selectable audio device.
#[derive(Clone, Debug)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Configuration for microphone capture.
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    /// Explicit input device name; `None` uses the host default.
    pub device_name: Option<String>,
    /// How many native-rate chunks may queue before samples are dropped.
    pub queue_chunks: usize,
    /// Requested sample rate. The device may negotiate a different one; the
    /// actual rate is reported in [`AudioCapture::native_sample_rate`].
    pub desired_sample_rate: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            queue_chunks: 64,
            desired_sample_rate: 16_000,
        }
    }
}

/// An open microphone stream feeding mono `f32` PCM into a bounded queue.
///
/// The stream is owned by this value; dropping it closes the device. The
/// callback downmixes to mono and uses `try_send`, so it never blocks.
pub struct AudioCapture {
    _stream: cpal::Stream,
    rx: Receiver<Vec<f32>>,
    pub native_sample_rate: u32,
    pub native_channels: u16,
    dropped_chunks: Arc<AtomicU64>,
}

impl AudioCapture {
    /// Open the default (or named) input device.
    pub fn open(config: &CaptureConfig) -> Result<Self, SpeechError> {
        let host = cpal::default_host();
        let device = match &config.device_name {
            Some(name) => host
                .input_devices()
                .map_err(|e| SpeechError::Device(e.to_string()))?
                .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
                .ok_or_else(|| SpeechError::Device(format!("no input device named '{name}'")))?,
            None => host
                .default_input_device()
                .ok_or_else(|| SpeechError::Device("no default input device".into()))?,
        };
        let supported = device
            .default_input_config()
            .map_err(|e| SpeechError::Device(e.to_string()))?;
        let sample_format = supported.sample_format();
        let stream_config: StreamConfig = supported.into();
        let channels = stream_config.channels;
        let rate = stream_config.sample_rate.0;
        let (tx, rx) = sync_channel::<Vec<f32>>(config.queue_chunks.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let dropped_cb = dropped.clone();

        let err_fn = |e| eprintln!("[audio] input stream error: {e}");
        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| push_mono(data, channels, &tx, &dropped_cb),
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let mono: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    push_mono(&mono, channels, &tx, &dropped_cb)
                },
                err_fn,
                None,
            ),
            SampleFormat::U16 => device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    let mono: Vec<f32> = data
                        .iter()
                        .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect();
                    push_mono(&mono, channels, &tx, &dropped_cb)
                },
                err_fn,
                None,
            ),
            other => {
                return Err(SpeechError::Device(format!(
                    "unsupported input sample format {other:?}"
                )))
            }
        }
        .map_err(|e| SpeechError::Device(e.to_string()))?;
        stream.play().map_err(|e| SpeechError::Device(e.to_string()))?;
        Ok(Self {
            _stream: stream,
            rx,
            native_sample_rate: rate,
            native_channels: channels,
            dropped_chunks: dropped,
        })
    }

    /// Blocking receive of the next mono chunk. Returns `None` when the device
    /// stream has ended.
    pub fn recv(&self) -> Option<Vec<f32>> {
        self.rx.recv().ok()
    }

    /// Non-blocking receive.
    pub fn try_recv(&self) -> Option<Vec<f32>> {
        self.rx.try_recv().ok()
    }

    pub fn dropped_chunks(&self) -> u64 {
        self.dropped_chunks.load(Ordering::Relaxed)
    }

    /// List available input devices.
    pub fn list_devices() -> Result<Vec<AudioDeviceInfo>, SpeechError> {
        let host = cpal::default_host();
        let default_name = host.default_input_device().and_then(|d| d.name().ok());
        let mut out = Vec::new();
        for d in host.input_devices().map_err(|e| SpeechError::Device(e.to_string()))? {
            if let Ok(name) = d.name() {
                let is_default = Some(&name) == default_name.as_ref();
                out.push(AudioDeviceInfo { name, is_default });
            }
        }
        Ok(out)
    }
}

fn push_mono(data: &[f32], channels: u16, tx: &SyncSender<Vec<f32>>, dropped: &AtomicU64) {
    let mono: Vec<f32> = if channels <= 1 {
        data.to_vec()
    } else {
        data.chunks(channels as usize)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect()
    };
    match tx.try_send(mono) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Disconnected(_)) => {}
    }
}

/// Configuration for speaker playback.
#[derive(Clone, Debug)]
pub struct PlaybackConfig {
    pub device_name: Option<String>,
    /// Sample rate of the PCM buffers that will be queued.
    pub sample_rate: u32,
    /// How many buffers may queue before the producer is told to wait.
    pub queue_buffers: usize,
}

impl PlaybackConfig {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            device_name: None,
            sample_rate,
            queue_buffers: 32,
        }
    }
}

/// An open speaker stream that plays queued mono `f32` buffers.
///
/// The audio callback pulls from a bounded channel; when nothing is queued it
/// emits silence and increments the underrun counter. Nothing blocks in the
/// callback.
pub struct AudioPlayback {
    _stream: cpal::Stream,
    tx: SyncSender<Vec<f32>>,
    underruns: Arc<AtomicU64>,
    queued: Arc<AtomicU64>,
    played: Arc<AtomicU64>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioPlayback {
    pub fn open(config: &PlaybackConfig) -> Result<Self, SpeechError> {
        let host = cpal::default_host();
        let device = match &config.device_name {
            Some(name) => host
                .output_devices()
                .map_err(|e| SpeechError::Device(e.to_string()))?
                .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
                .ok_or_else(|| SpeechError::Device(format!("no output device named '{name}'")))?,
            None => host
                .default_output_device()
                .ok_or_else(|| SpeechError::Device("no default output device".into()))?,
        };
        // Prefer a config matching the requested sample rate; otherwise take
        // the device default (playback then runs at the device rate and the
        // caller is told which rate to synthesize at).
        let supported = choose_output_config(&device, config.sample_rate)
            .ok_or_else(|| SpeechError::Device("no usable output config".into()))?;
        let sample_format = supported.sample_format();
        let stream_config: StreamConfig = supported.into();
        let channels = stream_config.channels;
        let rate = stream_config.sample_rate.0;
        let (tx, rx) = sync_channel::<Vec<f32>>(config.queue_buffers.max(1));
        let underruns = Arc::new(AtomicU64::new(0));
        let underruns_cb = underruns.clone();
        let played = Arc::new(AtomicU64::new(0));
        let played_cb = played.clone();
        let err_fn = |e| eprintln!("[audio] output stream error: {e}");

        let stream = match sample_format {
            SampleFormat::F32 => {
                let mut feeder = OutputFeeder::new(rx, played_cb);
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _| feeder.fill(data, channels, &underruns_cb),
                    err_fn,
                    None,
                )
            }
            SampleFormat::I16 => {
                let mut feeder = OutputFeeder::new(rx, played_cb);
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _| {
                        let mut mono = vec![0.0f32; data.len() / channels.max(1) as usize];
                        feeder.fill(&mut mono, channels, &underruns_cb);
                        for (dst, src) in data.iter_mut().zip(interleave(&mono, channels)) {
                            *dst = (src.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let mut feeder = OutputFeeder::new(rx, played_cb);
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [u16], _| {
                        let mut mono = vec![0.0f32; data.len() / channels.max(1) as usize];
                        feeder.fill(&mut mono, channels, &underruns_cb);
                        for (dst, src) in data.iter_mut().zip(interleave(&mono, channels)) {
                            let v = (src.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32;
                            *dst = v as u16;
                        }
                    },
                    err_fn,
                    None,
                )
            }
            other => {
                return Err(SpeechError::Device(format!(
                    "unsupported output sample format {other:?}"
                )))
            }
        }
        .map_err(|e| SpeechError::Device(e.to_string()))?;
        stream.play().map_err(|e| SpeechError::Device(e.to_string()))?;
        Ok(Self {
            _stream: stream,
            tx,
            underruns,
            queued: Arc::new(AtomicU64::new(0)),
            played,
            sample_rate: rate,
            channels,
        })
    }

    /// Queue a mono buffer for playback. Blocks briefly if the queue is full so
    /// the producer naturally applies backpressure instead of dropping audio.
    pub fn queue(&self, samples: &[f32]) -> Result<(), SpeechError> {
        self.tx
            .send(samples.to_vec())
            .map_err(|_| SpeechError::Device("playback stream closed".into()))?;
        self.queued.fetch_add(samples.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Queue without blocking; returns `false` if the queue is full.
    pub fn try_queue(&self, samples: &[f32]) -> bool {
        if self.tx.try_send(samples.to_vec()).is_ok() {
            self.queued.fetch_add(samples.len() as u64, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Total mono samples handed to the device for playback.
    pub fn queued_samples(&self) -> u64 {
        self.queued.load(Ordering::Relaxed)
    }

    /// Total mono samples the device callback has consumed so far.
    pub fn played_samples(&self) -> u64 {
        self.played.load(Ordering::Relaxed)
    }

    /// Block until every queued sample has been consumed by the device, or
    /// until `timeout` elapses. Returns `true` if the queue drained.
    ///
    /// This is the honest way to know that audio really finished playing;
    /// sleeping for the nominal duration would be a guess.
    pub fn wait_until_drained(&self, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while self.played_samples() < self.queued_samples() {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        true
    }

    pub fn underruns(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }

    pub fn list_devices() -> Result<Vec<AudioDeviceInfo>, SpeechError> {
        let host = cpal::default_host();
        let default_name = host.default_output_device().and_then(|d| d.name().ok());
        let mut out = Vec::new();
        for d in host.output_devices().map_err(|e| SpeechError::Device(e.to_string()))? {
            if let Ok(name) = d.name() {
                let is_default = Some(&name) == default_name.as_ref();
                out.push(AudioDeviceInfo { name, is_default });
            }
        }
        Ok(out)
    }
}

fn choose_output_config(
    device: &cpal::Device,
    desired_rate: u32,
) -> Option<cpal::SupportedStreamConfig> {
    if let Ok(configs) = device.supported_output_configs() {
        for c in configs {
            if c.min_sample_rate().0 <= desired_rate && desired_rate <= c.max_sample_rate().0 {
                return Some(c.with_sample_rate(cpal::SampleRate(desired_rate)));
            }
        }
    }
    device.default_output_config().ok()
}

fn interleave(mono: &[f32], channels: u16) -> impl Iterator<Item = f32> + '_ {
    let ch = channels.max(1) as usize;
    (0..mono.len() * ch).map(move |i| mono[i / ch])
}

/// Pulls queued mono buffers into device callbacks.
///
/// A single cpal callback often asks for fewer frames than one queued buffer
/// (or more), so this keeps a carry-over position instead of truncating. That
/// makes queueing a whole utterance correct: every sample is eventually played,
/// in order, exactly once.
struct OutputFeeder {
    rx: Receiver<Vec<f32>>,
    current: Vec<f32>,
    pos: usize,
    played: Arc<AtomicU64>,
}

impl OutputFeeder {
    fn new(rx: Receiver<Vec<f32>>, played: Arc<AtomicU64>) -> Self {
        Self {
            rx,
            current: Vec::new(),
            pos: 0,
            played,
        }
    }

    /// Fill `data` (interleaved, `channels` wide) with the next mono samples.
    /// Any frames that run past the end of the queued audio are silent.
    fn fill(&mut self, data: &mut [f32], channels: u16, underruns: &AtomicU64) {
        let ch = channels.max(1) as usize;
        let frames = data.len() / ch;
        let mut written = 0;
        while written < frames {
            if self.pos >= self.current.len() {
                match self.rx.try_recv() {
                    Ok(buf) => {
                        self.current = buf;
                        self.pos = 0;
                    }
                    Err(_) => break,
                }
            }
            let avail = self.current.len() - self.pos;
            let take = avail.min(frames - written);
            for i in 0..take {
                let v = self.current[self.pos + i];
                let base = (written + i) * ch;
                for s in data[base..base + ch].iter_mut() {
                    *s = v;
                }
            }
            self.pos += take;
            written += take;
        }
        if written < frames {
            underruns.fetch_add(1, Ordering::Relaxed);
        }
        for s in data[written * ch..].iter_mut() {
            *s = 0.0;
        }
        self.played.fetch_add(written as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod feeder_tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn feeder_carries_leftover_across_callbacks() {
        let (tx, rx) = sync_channel::<Vec<f32>>(4);
        tx.send(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let played = Arc::new(AtomicU64::new(0));
        let mut feeder = OutputFeeder::new(rx, played.clone());
        let underruns = AtomicU64::new(0);
        // First callback wants only 2 mono frames.
        let mut d1 = [0.0f32; 2];
        feeder.fill(&mut d1, 1, &underruns);
        assert_eq!(d1, [1.0, 2.0]);
        // Second callback wants 4 frames; 3 remain then silence.
        let mut d2 = [0.0f32; 4];
        feeder.fill(&mut d2, 1, &underruns);
        assert_eq!(d2, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(played.load(Ordering::Relaxed), 5);
        assert_eq!(underruns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn feeder_reports_underrun_only_when_starved() {
        let (_tx, rx) = sync_channel::<Vec<f32>>(4);
        let played = Arc::new(AtomicU64::new(0));
        let mut feeder = OutputFeeder::new(rx, played.clone());
        let underruns = AtomicU64::new(0);
        let mut d = [0.0f32; 8];
        feeder.fill(&mut d, 1, &underruns);
        assert!(d.iter().all(|&v| v == 0.0));
        assert_eq!(played.load(Ordering::Relaxed), 0);
        assert_eq!(underruns.load(Ordering::Relaxed), 1);
    }
}

/// A small linear resampler for converting captured audio to a target rate.
///
/// Deliberately simple and deterministic; adequate for 44.1/48 kHz → 16 kHz
/// downsampling ahead of VAD/ASR. This is not a high-fidelity resampler.
#[derive(Clone, Debug)]
pub struct LinearResampler {
    from: u32,
    to: u32,
    pos: f64,
    last: f32,
}

impl LinearResampler {
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            from,
            to,
            pos: 0.0,
            last: 0.0,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from == self.to || input.is_empty() {
            return input.to_vec();
        }
        let ratio = self.from as f64 / self.to as f64;
        let mut out = Vec::with_capacity((input.len() as f64 / ratio) as usize + 2);
        let mut pos = self.pos;
        while pos < input.len() as f64 {
            let i = pos.floor() as usize;
            let frac = (pos - i as f64) as f32;
            let a = if i == 0 { self.last } else { input[i - 1] };
            let b = input.get(i).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
            pos += ratio;
        }
        self.pos = pos - input.len() as f64;
        self.last = *input.last().unwrap();
        out
    }

    pub fn reset(&mut self) {
        self.pos = 0.0;
        self.last = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_downsamples_length_proportionally() {
        let mut r = LinearResampler::new(48_000, 16_000);
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = r.process(&input);
        // 4800 in at 48k -> ~1600 out at 16k.
        assert!((out.len() as i32 - 1600).abs() < 4, "len={}", out.len());
    }

    #[test]
    fn resampler_passthrough_at_same_rate() {
        let mut r = LinearResampler::new(16_000, 16_000);
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(r.process(&input), input);
    }

    #[test]
    fn interleave_repeats_channels() {
        let v: Vec<f32> = interleave(&[1.0, 2.0], 2).collect();
        assert_eq!(v, vec![1.0, 1.0, 2.0, 2.0]);
    }
}
