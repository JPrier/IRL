//! Device-independent audio signal processing: downmixing, sample-rate
//! conversion and level metering.
//!
//! ## Audio data contract
//!
//! Every stage after the device callback works on **mono `f32` PCM in
//! `[-1, 1]`** whose sample rate travels with it. Interleaved multichannel
//! data exists only inside a device callback, where [`downmix_into`] converts
//! it exactly once. Nothing downstream may downmix again: doing so halves the
//! sample count of already-mono audio, which plays it back at double speed
//! and makes speech unrecognisable (the original bevy-game capture bug).
//!
//! This module has no native dependencies, so it is always compiled and is
//! fully testable on machines without audio devices or speech models.

/// Average one interleaved frame per output sample.
///
/// `convert` maps a native sample to `f32` in `[-1, 1]`. The output is cleared
/// first and receives exactly `data.len() / channels` samples; a trailing
/// partial frame (which a well-behaved device never produces) is dropped.
pub fn downmix_into<T: Copy>(
    data: &[T],
    channels: u16,
    convert: impl Fn(T) -> f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    let ch = channels.max(1) as usize;
    if ch == 1 {
        out.extend(data.iter().map(|s| convert(*s)));
        return;
    }
    let scale = 1.0 / ch as f32;
    out.extend(
        data.chunks_exact(ch)
            .map(|frame| frame.iter().map(|s| convert(*s)).sum::<f32>() * scale),
    );
}

/// Convenience wrapper around [`downmix_into`] for `f32` input.
pub fn downmix(data: &[f32], channels: u16) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / channels.max(1) as usize);
    downmix_into(data, channels, |s| s, &mut out);
    out
}

/// `i16` PCM to `f32` in `[-1, 1)`.
pub fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

/// Unsigned 16-bit PCM (offset binary) to `f32` in `[-1, 1)`.
pub fn u16_to_f32(s: u16) -> f32 {
    (s as f32 - 32768.0) / 32768.0
}

/// RMS and peak of a block of samples.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Level {
    pub rms: f32,
    pub peak: f32,
}

impl Level {
    pub fn of(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sum = 0.0f64;
        let mut peak = 0.0f32;
        for s in samples {
            sum += (*s as f64) * (*s as f64);
            peak = peak.max(s.abs());
        }
        Self {
            rms: (sum / samples.len() as f64).sqrt() as f32,
            peak,
        }
    }

    /// RMS in dBFS, floored at -100 dB for silence.
    pub fn rms_dbfs(&self) -> f32 {
        if self.rms <= 1e-5 {
            -100.0
        } else {
            20.0 * self.rms.log10()
        }
    }
}

/// A streaming low-pass FIR filter (windowed sinc). It carries its history
/// across calls, so filtering a signal in chunks is bit-identical to filtering
/// it in one piece.
#[derive(Clone, Debug)]
struct FirLowpass {
    taps: Vec<f32>,
    history: Vec<f32>,
}

impl FirLowpass {
    /// `cutoff` is in cycles per sample (0 < cutoff < 0.5).
    fn new(cutoff: f64, len: usize) -> Self {
        let len = len | 1; // odd length: symmetric, integer group delay
        let mid = (len / 2) as f64;
        let mut taps: Vec<f64> = (0..len)
            .map(|n| {
                let x = n as f64 - mid;
                let sinc = if x == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                // Blackman window.
                let w = 0.42
                    - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (len - 1) as f64).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * n as f64 / (len - 1) as f64).cos();
                sinc * w
            })
            .collect();
        // Unity gain at DC.
        let sum: f64 = taps.iter().sum();
        for t in &mut taps {
            *t /= sum;
        }
        Self {
            taps: taps.into_iter().map(|t| t as f32).collect(),
            history: vec![0.0; len - 1],
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(input.len());
        let n = self.taps.len();
        // Work over [history | input] without allocating per sample.
        self.history.extend_from_slice(input);
        for i in 0..input.len() {
            let window = &self.history[i..i + n];
            let acc: f32 = window
                .iter()
                .zip(self.taps.iter().rev())
                .map(|(x, t)| x * t)
                .sum();
            out.push(acc);
        }
        let keep = self.history.len() - (n - 1);
        self.history.drain(..keep);
    }

    fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = 0.0);
    }
}

/// Streaming sample-rate converter for mono PCM.
///
/// * Output sample `k` sits at input position `k * from / to`, computed in
///   exact integer arithmetic, so the result does **not** depend on how the
///   input is chunked and long streams never drift.
/// * Downsampling first applies an anti-aliasing low-pass filter (44.1/48 kHz
///   microphones carry energy above the 8 kHz Nyquist limit of 16 kHz ASR).
/// * Over `n` input samples it emits `n * to / from` samples to within one
///   (an output that needs a not-yet-received input sample waits for the
///   next chunk), so duration is preserved across any chunking.
#[derive(Clone, Debug)]
pub struct LinearResampler {
    from: u32,
    to: u32,
    /// Index of the next output sample.
    next_out: u64,
    /// Global index of the first sample in the pending window.
    base: u64,
    /// The last sample of the previous chunk (global index `base - 1`).
    prev: f32,
    lowpass: Option<FirLowpass>,
    scratch: Vec<f32>,
}

impl LinearResampler {
    pub fn new(from: u32, to: u32) -> Self {
        let from = from.max(1);
        let to = to.max(1);
        let lowpass = (to < from).then(|| {
            // Pass band up to 90% of the output Nyquist frequency.
            let cutoff = 0.5 * 0.9 * to as f64 / from as f64;
            FirLowpass::new(cutoff, 63)
        });
        Self {
            from,
            to,
            next_out: 0,
            base: 0,
            prev: 0.0,
            lowpass,
            scratch: Vec::new(),
        }
    }

    pub fn from_rate(&self) -> u32 {
        self.from
    }

    pub fn to_rate(&self) -> u32 {
        self.to
    }

    /// Convert one chunk. Chunks must be consecutive pieces of one stream.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.process_into(input, &mut out);
        out
    }

    /// Like [`LinearResampler::process`], appending to `out`.
    pub fn process_into(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.from == self.to {
            out.extend_from_slice(input);
            return;
        }
        if input.is_empty() {
            return;
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        let x: &[f32] = match self.lowpass.as_mut() {
            Some(lp) => {
                lp.process(input, &mut scratch);
                &scratch
            }
            None => input,
        };
        let len = x.len() as u64;
        let (from, to) = (self.from as u64, self.to as u64);
        out.reserve((len * to / from) as usize + 1);
        let (base, prev) = (self.base, self.prev);
        let sample = |g: u64| -> f32 {
            if g + 1 == base {
                prev
            } else {
                x[(g - base) as usize]
            }
        };
        loop {
            let num = self.next_out * from;
            let idx = num / to; // global index of the left neighbour
            let frac = (num % to) as f32 / to as f32;
            // Need samples idx and idx + 1 (when frac > 0).
            let right = if frac > 0.0 { idx + 1 } else { idx };
            if right >= base + len {
                break;
            }
            let a = sample(idx);
            let v = if frac > 0.0 {
                a + (sample(idx + 1) - a) * frac
            } else {
                a
            };
            out.push(v);
            self.next_out += 1;
        }
        self.prev = x[x.len() - 1];
        self.base += len;
        self.scratch = scratch;
    }

    /// Forget all history so the next chunk starts a new stream.
    pub fn reset(&mut self) {
        self.next_out = 0;
        self.base = 0;
        self.prev = 0.0;
        if let Some(lp) = self.lowpass.as_mut() {
            lp.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn stereo_downmix_averages_frames_once() {
        assert_eq!(downmix(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(downmix(&[0.5, 1.0], 1), vec![0.5, 1.0]);
        // 48 kHz stereo for 10 ms is 480 frames -> 480 mono samples, not 240.
        let interleaved = vec![0.25f32; 960];
        assert_eq!(downmix(&interleaved, 2).len(), 480);
    }

    #[test]
    fn integer_formats_convert_to_unit_range() {
        assert_eq!(i16_to_f32(0), 0.0);
        assert!((i16_to_f32(i16::MIN) + 1.0).abs() < 1e-6);
        assert!((u16_to_f32(32768)).abs() < 1e-6);
        assert!((u16_to_f32(0) + 1.0).abs() < 1e-6);
        let mut out = Vec::new();
        downmix_into(&[i16::MAX, i16::MAX, 0, 0], 2, i16_to_f32, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.99 && out[1] == 0.0);
    }

    #[test]
    fn level_reports_rms_and_peak() {
        let l = Level::of(&[0.5, -0.5, 0.5, -0.5]);
        assert!((l.rms - 0.5).abs() < 1e-6);
        assert!((l.peak - 0.5).abs() < 1e-6);
        assert_eq!(Level::of(&[]).rms_dbfs(), -100.0);
    }

    #[test]
    fn common_mic_rates_convert_to_16k_with_correct_duration() {
        for from in [44_100u32, 48_000, 32_000, 22_050, 16_000, 8_000] {
            let input = tone(from, 440.0, 2.0);
            let mut r = LinearResampler::new(from, 16_000);
            let out = r.process(&input);
            let expected = input.len() as f64 * 16_000.0 / from as f64;
            assert!(
                (out.len() as f64 - expected).abs() <= 1.0,
                "{from} Hz: {} vs {expected}",
                out.len()
            );
        }
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let input = tone(44_100, 300.0, 1.0);
        let mut whole = LinearResampler::new(44_100, 16_000);
        let a = whole.process(&input);
        let mut chunked = LinearResampler::new(44_100, 16_000);
        let mut b = Vec::new();
        // Irregular chunk sizes, as a device delivers them.
        let mut i = 0;
        for (n, size) in [1usize, 7, 441, 512, 1024, 3, 480]
            .iter()
            .cycle()
            .enumerate()
        {
            if i >= input.len() {
                break;
            }
            let end = (i + size + n % 5).min(input.len());
            chunked.process_into(&input[i..end], &mut b);
            i = end;
        }
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).all(|(x, y)| x == y),
            "chunked output differs"
        );
    }

    #[test]
    fn downsampling_preserves_speech_band_and_rejects_aliases() {
        // 1 kHz (speech band) must pass; 12 kHz (above the 8 kHz Nyquist of
        // the 16 kHz output) must be strongly attenuated instead of folding
        // back to 4 kHz.
        let mut r = LinearResampler::new(48_000, 16_000);
        let pass = Level::of(&r.process(&tone(48_000, 1_000.0, 0.5))[200..]);
        let mut r = LinearResampler::new(48_000, 16_000);
        let stop = Level::of(&r.process(&tone(48_000, 12_000.0, 0.5))[200..]);
        assert!(pass.rms > 0.3, "pass band rms {}", pass.rms);
        assert!(stop.rms < 0.01, "alias rms {}", stop.rms);
    }

    #[test]
    fn reset_starts_a_new_stream() {
        let input = tone(48_000, 200.0, 0.1);
        let mut r = LinearResampler::new(48_000, 16_000);
        let a = r.process(&input);
        r.reset();
        let b = r.process(&input);
        assert_eq!(a, b);
    }

    #[test]
    fn passthrough_at_same_rate() {
        let mut r = LinearResampler::new(16_000, 16_000);
        assert_eq!(r.process(&[0.1, 0.2, 0.3]), vec![0.1, 0.2, 0.3]);
    }
}
