//! Real audio-device I/O: microphone capture and speaker playback via `cpal`.
//!
//! Design constraints (see the design brief):
//! * audio callbacks are **nonblocking** and never run inference or disk I/O;
//! * buffers are bounded, so a slow consumer drops audio in a controlled way
//!   (and counts it) rather than growing memory without limit;
//! * the capture callback downmixes to mono exactly once, with one allocation
//!   per device buffer; resampling and metering happen on the consumer side.
//!
//! ## Capture data contract
//!
//! [`AudioCapture`] delivers **mono `f32` chunks at
//! [`AudioCapture::native_sample_rate`]**. The device may be stereo or wider
//! ([`AudioCapture::device_channels`]), but that is informational only: the
//! chunks are already downmixed and must never be downmixed again. See
//! [`crate::dsp`] for the rest of the contract.
//!
//! ## Playback epochs
//!
//! Each queued playback buffer carries an *epoch*. When the shared epoch
//! counter ([`PlaybackConfig::epoch`]) moves past a buffer's epoch, the device
//! callback discards it — including a buffer already half played — so a
//! cancelled or superseded line goes silent within one device period without
//! the producer having to reach into the queue.
//!
//! On machines without a microphone or output device, construction fails with
//! [`SpeechError::Device`]; the WAV-fixture paths remain fully testable without
//! any device at all.

use rpg_domain::ports::SpeechError;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

pub use crate::dsp::LinearResampler;
use crate::dsp::{downmix_into, i16_to_f32, u16_to_f32};

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
    /// How many device buffers may queue before samples are dropped.
    pub queue_chunks: usize,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            queue_chunks: 256,
        }
    }
}

/// Latest asynchronous error reported by a device stream.
#[derive(Clone, Default)]
struct StreamFault {
    flag: Arc<AtomicBool>,
    message: Arc<Mutex<Option<String>>>,
}

impl StreamFault {
    fn report(&self, message: String) {
        if let Ok(mut m) = self.message.lock() {
            *m = Some(message);
        }
        self.flag.store(true, Ordering::Release);
    }

    fn take(&self) -> Option<String> {
        if !self.flag.swap(false, Ordering::AcqRel) {
            return None;
        }
        self.message.lock().ok().and_then(|mut m| m.take())
    }
}

/// An open microphone stream feeding mono `f32` PCM into a bounded queue.
///
/// The stream is owned by this value; dropping it closes the device. Some
/// platforms require the stream to stay on the thread that opened it, so the
/// realtime client opens and owns it on a dedicated capture thread.
pub struct AudioCapture {
    _stream: cpal::Stream,
    rx: Receiver<Vec<f32>>,
    /// Sample rate of every delivered chunk, in Hz.
    pub native_sample_rate: u32,
    /// Channel count of the device stream. Informational only: delivered
    /// chunks are already mono.
    pub device_channels: u16,
    /// Native sample format of the device stream.
    pub sample_format: String,
    pub device_name: String,
    dropped_chunks: Arc<AtomicU64>,
    fault: StreamFault,
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
            None => host.default_input_device().ok_or_else(|| {
                SpeechError::Device(
                    "no default input device (is a microphone connected and allowed?)".into(),
                )
            })?,
        };
        let device_name = device.name().unwrap_or_else(|_| "<unnamed>".into());
        let supported = device.default_input_config().map_err(|e| {
            SpeechError::Device(format!(
                "cannot query '{device_name}' (microphone permission?): {e}"
            ))
        })?;
        let sample_format = supported.sample_format();
        let stream_config: StreamConfig = supported.into();
        let channels = stream_config.channels;
        let rate = stream_config.sample_rate.0;
        let (tx, rx) = sync_channel::<Vec<f32>>(config.queue_chunks.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let fault = StreamFault::default();

        let err_fault = fault.clone();
        let err_fn = move |e: cpal::StreamError| err_fault.report(e.to_string());
        let stream = match sample_format {
            SampleFormat::F32 => {
                let (tx, dropped) = (tx.clone(), dropped.clone());
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _| push_mono(data, channels, |s| s, &tx, &dropped),
                    err_fn,
                    None,
                )
            }
            SampleFormat::I16 => {
                let (tx, dropped) = (tx.clone(), dropped.clone());
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _| push_mono(data, channels, i16_to_f32, &tx, &dropped),
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let (tx, dropped) = (tx.clone(), dropped.clone());
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _| push_mono(data, channels, u16_to_f32, &tx, &dropped),
                    err_fn,
                    None,
                )
            }
            other => {
                return Err(SpeechError::Device(format!(
                    "unsupported input sample format {other:?}"
                )))
            }
        }
        .map_err(|e| SpeechError::Device(format!("cannot open '{device_name}': {e}")))?;
        stream
            .play()
            .map_err(|e| SpeechError::Device(format!("cannot start '{device_name}': {e}")))?;
        Ok(Self {
            _stream: stream,
            rx,
            native_sample_rate: rate,
            device_channels: channels,
            sample_format: format!("{sample_format:?}"),
            device_name,
            dropped_chunks: dropped,
            fault,
        })
    }

    /// Blocking receive of the next mono chunk. Returns `None` when the device
    /// stream has ended. Never call this from a realtime or simulation thread.
    pub fn recv(&self) -> Option<Vec<f32>> {
        self.rx.recv().ok()
    }

    /// Non-blocking receive of the next mono chunk.
    pub fn try_recv(&self) -> Option<Vec<f32>> {
        self.rx.try_recv().ok()
    }

    /// Discard everything already queued (audio from before the caller started
    /// listening). Returns the number of samples dropped.
    pub fn clear(&self) -> usize {
        let mut n = 0;
        while let Ok(chunk) = self.rx.try_recv() {
            n += chunk.len();
        }
        n
    }

    /// Device buffers lost because the queue was full.
    pub fn dropped_chunks(&self) -> u64 {
        self.dropped_chunks.load(Ordering::Relaxed)
    }

    /// The most recent stream error (device unplugged, permission revoked…),
    /// if one occurred since the last call.
    pub fn take_error(&self) -> Option<String> {
        self.fault.take()
    }

    /// List available input devices.
    pub fn list_devices() -> Result<Vec<AudioDeviceInfo>, SpeechError> {
        let host = cpal::default_host();
        let default_name = host.default_input_device().and_then(|d| d.name().ok());
        let mut out = Vec::new();
        for d in host
            .input_devices()
            .map_err(|e| SpeechError::Device(e.to_string()))?
        {
            if let Ok(name) = d.name() {
                let is_default = Some(&name) == default_name.as_ref();
                out.push(AudioDeviceInfo { name, is_default });
            }
        }
        Ok(out)
    }
}

/// Device-callback body: downmix once (a single allocation) and hand the chunk
/// off without blocking. A full queue drops the chunk and counts it.
fn push_mono<T: Copy>(
    data: &[T],
    channels: u16,
    convert: impl Fn(T) -> f32,
    tx: &SyncSender<Vec<f32>>,
    dropped: &AtomicU64,
) {
    let mut mono = Vec::with_capacity(data.len() / channels.max(1) as usize);
    downmix_into(data, channels, convert, &mut mono);
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
    /// How many buffers may queue before submission reports "full".
    pub queue_buffers: usize,
    /// Shared playback epoch. Buffers queued with an older epoch are dropped
    /// by the device callback. `None` creates a private counter that stays 0.
    pub epoch: Option<Arc<AtomicU64>>,
}

impl PlaybackConfig {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            device_name: None,
            sample_rate,
            queue_buffers: 256,
            epoch: None,
        }
    }
}

/// Counters shared between the playback handle and the device callback.
#[derive(Default)]
struct PlaybackCounters {
    queued: AtomicU64,
    played: AtomicU64,
    discarded: AtomicU64,
    underruns: AtomicU64,
    /// Set by the producer while an utterance is still being generated, so
    /// running dry in the middle of it counts as an underrun.
    streaming: AtomicBool,
}

/// An open speaker stream that plays queued mono `f32` buffers.
///
/// The audio callback pulls from a bounded channel; when nothing is queued it
/// emits silence and counts an underrun. Nothing blocks in the callback, and
/// [`AudioPlayback::try_submit`] never blocks the producer either.
pub struct AudioPlayback {
    _stream: cpal::Stream,
    tx: SyncSender<(u64, Vec<f32>)>,
    counters: Arc<PlaybackCounters>,
    epoch: Arc<AtomicU64>,
    fault: StreamFault,
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
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
        let device_name = device.name().unwrap_or_else(|_| "<unnamed>".into());
        // Prefer a config matching the requested sample rate; otherwise take
        // the device default (playback then runs at the device rate and the
        // caller is told which rate to synthesize at).
        let supported = choose_output_config(&device, config.sample_rate)
            .ok_or_else(|| SpeechError::Device("no usable output config".into()))?;
        let sample_format = supported.sample_format();
        let stream_config: StreamConfig = supported.into();
        let channels = stream_config.channels;
        let rate = stream_config.sample_rate.0;
        let (tx, rx) = sync_channel::<(u64, Vec<f32>)>(config.queue_buffers.max(1));
        let counters = Arc::new(PlaybackCounters::default());
        let epoch = config
            .epoch
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicU64::new(0)));
        let fault = StreamFault::default();
        let err_fault = fault.clone();
        let err_fn = move |e: cpal::StreamError| err_fault.report(e.to_string());
        let mut feeder = OutputFeeder::new(rx, counters.clone(), epoch.clone());

        let stream = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| feeder.fill(data, channels),
                err_fn,
                None,
            ),
            SampleFormat::I16 => {
                let mut scratch: Vec<f32> = Vec::new();
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _| {
                        scratch.resize(data.len(), 0.0);
                        feeder.fill(&mut scratch, channels);
                        for (dst, src) in data.iter_mut().zip(&scratch) {
                            *dst = (src.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let mut scratch: Vec<f32> = Vec::new();
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [u16], _| {
                        scratch.resize(data.len(), 0.0);
                        feeder.fill(&mut scratch, channels);
                        for (dst, src) in data.iter_mut().zip(&scratch) {
                            *dst = ((src.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
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
        .map_err(|e| SpeechError::Device(format!("cannot open '{device_name}': {e}")))?;
        stream
            .play()
            .map_err(|e| SpeechError::Device(format!("cannot start '{device_name}': {e}")))?;
        Ok(Self {
            _stream: stream,
            tx,
            counters,
            epoch,
            fault,
            sample_rate: rate,
            channels,
            device_name,
        })
    }

    /// The epoch new buffers are tagged with by [`AudioPlayback::queue`].
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Queue a mono buffer for playback at the current epoch. Blocks briefly
    /// if the queue is full. For offline tools only: realtime code must use
    /// [`AudioPlayback::try_submit`].
    pub fn queue(&self, samples: &[f32]) -> Result<(), SpeechError> {
        let len = samples.len() as u64;
        self.tx
            .send((self.current_epoch(), samples.to_vec()))
            .map_err(|_| SpeechError::Device("playback stream closed".into()))?;
        self.counters.queued.fetch_add(len, Ordering::Relaxed);
        Ok(())
    }

    /// Queue without blocking; returns `false` if the queue is full.
    pub fn try_queue(&self, samples: &[f32]) -> bool {
        self.try_submit(self.current_epoch(), samples.to_vec())
            .is_ok()
    }

    /// Submit a buffer tagged with `epoch` without blocking. A full queue
    /// hands the buffer back so the caller can retry later without copying.
    pub fn try_submit(&self, epoch: u64, samples: Vec<f32>) -> Result<(), Vec<f32>> {
        let len = samples.len() as u64;
        match self.tx.try_send((epoch, samples)) {
            Ok(()) => {
                self.counters.queued.fetch_add(len, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full((_, s))) | Err(TrySendError::Disconnected((_, s))) => Err(s),
        }
    }

    /// Total mono samples handed to the device for playback.
    pub fn queued_samples(&self) -> u64 {
        self.counters.queued.load(Ordering::Relaxed)
    }

    /// Total mono samples the device callback has actually played.
    pub fn played_samples(&self) -> u64 {
        self.counters.played.load(Ordering::Relaxed)
    }

    /// Samples dropped because their epoch was superseded.
    pub fn discarded_samples(&self) -> u64 {
        self.counters.discarded.load(Ordering::Relaxed)
    }

    /// Samples the callback has finished with, played or discarded. A buffer
    /// that ends at queued-sample index `n` is done once this reaches `n`.
    pub fn consumed_samples(&self) -> u64 {
        self.played_samples() + self.discarded_samples()
    }

    /// Block until every queued sample has been consumed by the device, or
    /// until `timeout` elapses. Returns `true` if the queue drained.
    ///
    /// For offline tools only: realtime code polls
    /// [`AudioPlayback::consumed_samples`] instead of blocking.
    pub fn wait_until_drained(&self, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while self.consumed_samples() < self.queued_samples() {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        true
    }

    pub fn underruns(&self) -> u64 {
        self.counters.underruns.load(Ordering::Relaxed)
    }

    /// Tell the device callback whether an utterance is mid-generation, so a
    /// producer that falls behind playback is reported as an underrun.
    pub fn set_streaming(&self, streaming: bool) {
        self.counters.streaming.store(streaming, Ordering::Relaxed);
    }

    /// The most recent stream error, if one occurred since the last call.
    pub fn take_error(&self) -> Option<String> {
        self.fault.take()
    }

    pub fn list_devices() -> Result<Vec<AudioDeviceInfo>, SpeechError> {
        let host = cpal::default_host();
        let default_name = host.default_output_device().and_then(|d| d.name().ok());
        let mut out = Vec::new();
        for d in host
            .output_devices()
            .map_err(|e| SpeechError::Device(e.to_string()))?
        {
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

/// Pulls queued mono buffers into device callbacks.
///
/// A single cpal callback often asks for fewer frames than one queued buffer
/// (or more), so this keeps a carry-over position instead of truncating. That
/// makes queueing a whole utterance correct: every sample is eventually played,
/// in order, exactly once — unless its epoch is superseded, in which case the
/// rest of it is discarded and counted.
struct OutputFeeder {
    rx: Receiver<(u64, Vec<f32>)>,
    current: Vec<f32>,
    current_epoch: u64,
    pos: usize,
    counters: Arc<PlaybackCounters>,
    epoch: Arc<AtomicU64>,
}

impl OutputFeeder {
    fn new(
        rx: Receiver<(u64, Vec<f32>)>,
        counters: Arc<PlaybackCounters>,
        epoch: Arc<AtomicU64>,
    ) -> Self {
        Self {
            rx,
            current: Vec::new(),
            current_epoch: 0,
            pos: 0,
            counters,
            epoch,
        }
    }

    /// Fill `data` (interleaved, `channels` wide) with the next mono samples.
    /// Any frames that run past the end of the queued audio are silent.
    fn fill(&mut self, data: &mut [f32], channels: u16) {
        let ch = channels.max(1) as usize;
        let frames = data.len() / ch;
        let live_epoch = self.epoch.load(Ordering::Acquire);
        let mut written = 0;
        let mut discarded = 0u64;
        while written < frames {
            if self.pos < self.current.len() && self.current_epoch < live_epoch {
                discarded += (self.current.len() - self.pos) as u64;
                self.pos = self.current.len();
            }
            if self.pos >= self.current.len() {
                match self.rx.try_recv() {
                    Ok((epoch, buf)) => {
                        self.current = buf;
                        self.current_epoch = epoch;
                        self.pos = 0;
                        continue;
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
        // Underruns only matter while an utterance is still being produced:
        // an idle device emitting silence is not starved.
        if written < frames && self.counters.streaming.load(Ordering::Relaxed) {
            self.counters.underruns.fetch_add(1, Ordering::Relaxed);
        }
        for s in data[written * ch..].iter_mut() {
            *s = 0.0;
        }
        self.counters
            .played
            .fetch_add(written as u64, Ordering::Relaxed);
        if discarded > 0 {
            self.counters
                .discarded
                .fetch_add(discarded, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod feeder_tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    type Tagged = SyncSender<(u64, Vec<f32>)>;

    fn feeder(cap: usize) -> (Tagged, OutputFeeder, Arc<AtomicU64>) {
        let (tx, rx) = sync_channel(cap);
        let epoch = Arc::new(AtomicU64::new(0));
        let f = OutputFeeder::new(rx, Arc::new(PlaybackCounters::default()), epoch.clone());
        (tx, f, epoch)
    }

    fn send(tx: &Tagged, f: &OutputFeeder, epoch: u64, s: Vec<f32>) {
        f.counters
            .queued
            .fetch_add(s.len() as u64, Ordering::Relaxed);
        tx.send((epoch, s)).unwrap();
    }

    #[test]
    fn feeder_carries_leftover_across_callbacks() {
        let (tx, mut feeder, _) = feeder(4);
        send(&tx, &feeder, 0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // First callback wants only 2 mono frames.
        let mut d1 = [0.0f32; 2];
        feeder.fill(&mut d1, 1);
        assert_eq!(d1, [1.0, 2.0]);
        // Second callback wants 4 frames; 3 remain then silence.
        let mut d2 = [0.0f32; 4];
        feeder.fill(&mut d2, 1);
        assert_eq!(d2, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(feeder.counters.played.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn idle_silence_is_not_an_underrun() {
        let (_tx, mut feeder, _) = feeder(4);
        let mut d = [0.0f32; 8];
        feeder.fill(&mut d, 1);
        assert!(d.iter().all(|&v| v == 0.0));
        assert_eq!(feeder.counters.played.load(Ordering::Relaxed), 0);
        assert_eq!(feeder.counters.underruns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn starving_mid_utterance_is_an_underrun() {
        let (tx, mut feeder, _) = feeder(4);
        send(&tx, &feeder, 0, vec![1.0, 2.0]);
        // More of the utterance is still being synthesized.
        feeder.counters.streaming.store(true, Ordering::Relaxed);
        let mut d = [0.0f32; 4];
        feeder.fill(&mut d, 1);
        assert_eq!(feeder.counters.underruns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn superseded_epoch_is_discarded_even_mid_buffer() {
        let (tx, mut feeder, epoch) = feeder(4);
        send(&tx, &feeder, 0, vec![1.0; 6]);
        let mut d = [0.0f32; 2];
        feeder.fill(&mut d, 1);
        assert_eq!(d, [1.0, 1.0]);
        // Cancel: the rest of the old line and anything queued behind it at
        // the old epoch must never reach the speaker.
        send(&tx, &feeder, 0, vec![1.0; 3]);
        epoch.store(1, Ordering::Release);
        send(&tx, &feeder, 1, vec![7.0, 8.0]);
        let mut d = [0.0f32; 4];
        feeder.fill(&mut d, 1);
        assert_eq!(d, [7.0, 8.0, 0.0, 0.0]);
        let c = &feeder.counters;
        assert_eq!(c.played.load(Ordering::Relaxed), 4);
        assert_eq!(c.discarded.load(Ordering::Relaxed), 7);
        // Everything queued is accounted for, so completion detection works.
        assert_eq!(
            c.played.load(Ordering::Relaxed) + c.discarded.load(Ordering::Relaxed),
            c.queued.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn stereo_output_duplicates_mono_frames() {
        let (tx, mut feeder, _) = feeder(4);
        send(&tx, &feeder, 0, vec![0.5, 0.25]);
        let mut d = [0.0f32; 4];
        feeder.fill(&mut d, 2);
        assert_eq!(d, [0.5, 0.5, 0.25, 0.25]);
    }
}
