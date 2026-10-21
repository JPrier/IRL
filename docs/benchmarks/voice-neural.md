# Neural voice benchmark — Milestone A evidence

This document records **real, measured** neural TTS and ASR numbers for the
offline voice stack. Unlike the earlier diagnostic-tone numbers, these come
from actual ONNX neural models running locally through `sherpa-onnx`. No cloud
service is contacted at any point.

## Host

| Item | Value |
|---|---|
| OS | Windows 11 (MINGW64 / MSYS2 shell) |
| CPU | 16 logical cores (development host) |
| Toolchain | `rustc 1.91.1`, `x86_64-pc-windows-msvc` |
| GPU | NOT RECORDED (all runs are CPU-only, `provider = "cpu"`) |
| RAM | NOT RECORDED |
| Threads | 4 worker threads for TTS/ASR |
| Date | October 2026 |

## Models and assets

| Bundle | Kind | Model | License | Size |
|---|---|---|---|---|
| `kokoro-en-v0_19` | TTS | Kokoro 82M, 11 English voices | Apache-2.0 | 319.6 MB |
| `whisper-tiny-en` | ASR | Whisper tiny.en, int8 ONNX | MIT | 118.1 MB |
| `silero-vad` | VAD | Silero VAD | MIT | 0.6 MB |

Asset URLs and SHA-256 checksums are pinned in
[`assets/speech/manifest.json`](../../assets/speech/manifest.json). `voice-demo
install` downloads, checksums and unpacks them; `voice-demo list` reports the
installed state and available audio devices.

## Reproduce

```bash
cargo run --release -p voice-demo -- install
cargo run --release -p voice-demo -- list
cargo run --release -p voice-demo -- bench
```

Full raw output is saved in [`voice-neural-bench.txt`](voice-neural-bench.txt).

## TTS — Kokoro 82M (`af_sarah`), CPU, 4 threads

| Text | audio (s) | synth (ms) | RTF |
|---|---:|---:|---:|
| "The ledger is behind the flour." | 2.04 | 807.0 | 2.53× |
| "I will give you the medicine if you tell me who signed the book." | 2.97 | 943.5 | 3.15× |
| "Come back after dark and we will talk about the shipment." | 3.06 | 918.8 | 3.33× |
| **Aggregate** | **8.07** | **2670** | **3.02×** |

* RTF here is `audio_seconds / wall_seconds`; **higher is better**.
* `first_audio_ms == synth_ms` because Kokoro is used in whole-utterance mode:
  the model does not expose incremental audio, so first-audio latency equals
  total synthesis time. Streaming TTS is NOT TESTED and remains future work.
* Output is 24 kHz mono PCM. A 4.54 s utterance had peak 0.401 and RMS 0.065
  (speech-like, not silent, not a tone).

## ASR — Whisper tiny.en int8, CPU, 4 threads

| Input | audio (s) | latency (ms) | speed |
|---|---:|---:|---:|
| `test_wavs/0.wav` (16 kHz) | 6.62 | 244.2 | 27.1× real time |

Transcript: "After early nightfall, the yellow lamps would light up here and
there the squalid quarter of the brothels." — matches the reference transcript.

A second fixture (`test_wavs/1.wav`) transcribed accurately in 577 ms, and the
8 kHz fixture (`test_wavs/8k.wav`) in 172 ms (with a minor name error:
"Hester-Prin" for "Hester Prynne", expected for tiny.en on downsampled audio).

## End-to-end roundtrip (WAV → ASR → neural TTS → WAV / speaker)

| Input | ASR | TTS | RTF |
|---|---|---|---|
| `test_wavs/1.wav` | 577 ms | 13.32 s audio in 3509 ms | 3.80× |
| `test_wavs/8k.wav` | 172 ms | 4.13 s audio in 1155 ms | 3.57× |

Playback was verified through the real output device via
`voice-demo roundtrip <wav> --play`; when the device runs at 192 kHz the audio is
resampled from 24 kHz so playback speed is correct.

## Honest caveats

* These numbers are from **one development host**. Steam Deck, low-end GPUs and
  other CPUs are NOT TESTED.
* The 0.8B LLM is **not** in this path yet: Milestone A uses a canned reply
  ("I heard you say: …"). Wiring ASR text into the resident LLM and speaking the
  model's answer is Milestone B.
* Kokoro TTS is whole-utterance; first-audio latency is not yet low enough for
  natural barge-in. VAD endpointing exists but conversational turn-taking is NOT
  TESTED.
* Only the `af_sarah` voice was measured for TTS latency. The other 10 Kokoro
  voices share the same model and are expected to be equivalent, but that is an
  assumption, not a measurement.
