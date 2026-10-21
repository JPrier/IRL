//! Offline speech-asset manifest, verification and installation.
//!
//! Every neural speech model used by the game is described by a pinned
//! [`AssetBundle`] in `assets/speech/manifest.json`: a download URL, a
//! SHA-256 checksum, a byte size, a license and an attribution. Nothing is
//! fetched from the network at gameplay time — installation is an explicit,
//! reproducible step.
//!
//! The module is deliberately dependency-light so that manifest parsing and
//! path resolution can be unit-tested without the native speech runtime or
//! network access. The actual download/extract paths are behind the `neural`
//! feature.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Kind of asset a bundle provides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetKind {
    Tts,
    Asr,
    Vad,
}

/// How a downloaded bundle is unpacked into the model cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveFormat {
    /// A plain file copied verbatim into the model cache.
    #[serde(rename = "raw")]
    Raw,
    /// A bzip2-compressed tar archive.
    #[serde(rename = "tar.bz2")]
    TarBz2,
}

/// Relative file locations (within the model cache) for a bundle.
///
/// A flat struct with optional fields keeps the JSON readable and lets one
/// struct describe TTS, ASR and VAD bundles.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BundleFiles {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voices: Option<String>,
    #[serde(default)]
    pub tokens: Option<String>,
    #[serde(default)]
    pub data_dir: Option<String>,
    #[serde(default)]
    pub encoder: Option<String>,
    #[serde(default)]
    pub decoder: Option<String>,
    #[serde(default)]
    pub vad_model: Option<String>,
}

/// A selectable neural voice inside a TTS bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceAsset {
    pub id: String,
    pub display_name: String,
    /// Backend speaker id passed to the TTS engine.
    pub sid: i32,
    pub language: String,
}

/// One installable, checksummed asset bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetBundle {
    pub id: String,
    pub kind: AssetKind,
    #[serde(default)]
    pub description: String,
    pub license: String,
    #[serde(default)]
    pub attribution: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    pub format: ArchiveFormat,
    #[serde(default)]
    pub extract_dir: Option<String>,
    #[serde(default)]
    pub files: BundleFiles,
    #[serde(default)]
    pub voices: Vec<VoiceAsset>,
}

impl AssetBundle {
    /// Directory (inside the model cache) this bundle is installed into.
    ///
    /// `Raw` bundles live directly in the model cache; archives live in their
    /// own subdirectory named by `extract_dir` (or the bundle id).
    pub fn install_dir_name(&self) -> &str {
        match self.format {
            ArchiveFormat::Raw => "",
            ArchiveFormat::TarBz2 => self.extract_dir.as_deref().unwrap_or(&self.id),
        }
    }
}

/// The parsed `manifest.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeechManifest {
    pub version: u32,
    #[serde(default)]
    pub description: String,
    pub bundles: Vec<AssetBundle>,
}

impl SpeechManifest {
    pub fn parse(json: &str) -> Result<Self, AssetError> {
        let manifest: SpeechManifest =
            serde_json::from_str(json).map_err(|e| AssetError::Manifest(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, AssetError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            AssetError::Manifest(format!("cannot read {}: {e}", path.as_ref().display()))
        })?;
        Self::parse(&text)
    }

    /// Reject obviously broken manifests early with actionable messages.
    pub fn validate(&self) -> Result<(), AssetError> {
        if self.version != 1 {
            return Err(AssetError::Manifest(format!(
                "unsupported manifest version {} (expected 1)",
                self.version
            )));
        }
        if self.bundles.is_empty() {
            return Err(AssetError::Manifest("manifest has no bundles".into()));
        }
        for b in &self.bundles {
            if b.id.is_empty() {
                return Err(AssetError::Manifest("bundle with empty id".into()));
            }
            if b.sha256.len() != 64 || !b.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(AssetError::Manifest(format!(
                    "bundle '{}' has an invalid sha256 (expected 64 hex chars)",
                    b.id
                )));
            }
            if b.url.is_empty() {
                return Err(AssetError::Manifest(format!(
                    "bundle '{}' has an empty url",
                    b.id
                )));
            }
            if b.kind == AssetKind::Tts && b.voices.is_empty() {
                return Err(AssetError::Manifest(format!(
                    "tts bundle '{}' declares no voices",
                    b.id
                )));
            }
        }
        Ok(())
    }

    pub fn bundle(&self, id: &str) -> Option<&AssetBundle> {
        self.bundles.iter().find(|b| b.id == id)
    }

    pub fn first_of_kind(&self, kind: AssetKind) -> Option<&AssetBundle> {
        self.bundles.iter().find(|b| b.kind == kind)
    }
}

/// Errors from manifest handling, verification or installation.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("manifest error: {0}")]
    Manifest(String),
    #[error("unknown asset bundle: {0}")]
    UnknownBundle(String),
    #[error("missing speech asset '{id}': expected {path} (run `voice-assets install {id}`)")]
    MissingAsset { id: String, path: String },
    #[error("checksum mismatch for '{id}': expected {expected}, found {found}")]
    Checksum {
        id: String,
        expected: String,
        found: String,
    },
    #[error("asset size mismatch for '{id}': expected {expected} bytes, found {found}")]
    Size {
        id: String,
        expected: u64,
        found: u64,
    },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("install error: {0}")]
    Install(String),
}

/// Resolved view over a manifest and a model cache directory.
///
/// This type does **not** hold the native engines; it only knows where files
/// are and whether they are present and intact. The neural backend consumes it.
#[derive(Clone, Debug)]
pub struct SpeechAssets {
    cache_dir: PathBuf,
    manifest: SpeechManifest,
}

impl SpeechAssets {
    /// Build a resolver for `cache_dir` from an already-parsed manifest.
    pub fn new(cache_dir: impl Into<PathBuf>, manifest: SpeechManifest) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            manifest,
        }
    }

    /// Load `manifest.json` and bind it to `cache_dir`.
    pub fn load(cache_dir: impl Into<PathBuf>, manifest_path: impl AsRef<Path>) -> Result<Self, AssetError> {
        Ok(Self::new(cache_dir, SpeechManifest::load(manifest_path)?))
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn manifest(&self) -> &SpeechManifest {
        &self.manifest
    }

    /// Directory that extracted bundles are unpacked into.
    pub fn models_dir(&self) -> PathBuf {
        self.cache_dir.join("models")
    }

    /// Absolute path of a bundle-relative file.
    ///
    /// Manifest paths are declared relative to the model cache directory and
    /// already include the bundle's install subdirectory, so this is a direct
    /// join.
    pub fn file_path(&self, bundle_id: &str, relative: &str) -> Result<PathBuf, AssetError> {
        if self.manifest.bundle(bundle_id).is_none() {
            return Err(AssetError::UnknownBundle(bundle_id.to_string()));
        }
        Ok(self.models_dir().join(relative))
    }

    /// True when every declared file for `bundle_id` exists on disk.
    pub fn is_installed(&self, bundle_id: &str) -> bool {
        let Some(bundle) = self.manifest.bundle(bundle_id) else {
            return false;
        };
        bundle.declared_files().iter().all(|rel| {
            let p = self.models_dir().join(rel);
            if bundle.is_dir_entry(rel) {
                p.is_dir()
            } else {
                p.is_file()
            }
        })
    }

    /// Resolve the file paths for a bundle, or return a clear missing-asset
    /// error listing what is absent.
    pub fn require_bundle(&self, bundle_id: &str) -> Result<ResolvedBundle, AssetError> {
        let bundle = self
            .manifest
            .bundle(bundle_id)
            .ok_or_else(|| AssetError::UnknownBundle(bundle_id.to_string()))?
            .clone();
        let mut resolved = ResolvedBundle {
            bundle: bundle.clone(),
            dir: self.models_dir(),
            files: BundleFiles::default(),
        };
        for rel in bundle.declared_files() {
            let direct = self.models_dir().join(&rel);
            let present = if bundle.is_dir_entry(&rel) {
                direct.is_dir()
            } else {
                direct.is_file()
            };
            if !present {
                return Err(AssetError::MissingAsset {
                    id: bundle.id.clone(),
                    path: direct.display().to_string(),
                });
            }
            resolved.assign(&rel, direct);
        }
        Ok(resolved)
    }
}

/// A bundle whose declared files have all been resolved to real paths.
#[derive(Clone, Debug)]
pub struct ResolvedBundle {
    pub bundle: AssetBundle,
    pub dir: PathBuf,
    pub files: BundleFiles,
}

impl ResolvedBundle {
    fn assign(&mut self, relative: &str, absolute: PathBuf) {
        let s = absolute.display().to_string();
        if self.bundle.files.model.as_deref() == Some(relative) {
            self.files.model = Some(s);
        } else if self.bundle.files.voices.as_deref() == Some(relative) {
            self.files.voices = Some(s);
        } else if self.bundle.files.tokens.as_deref() == Some(relative) {
            self.files.tokens = Some(s);
        } else if self.bundle.files.data_dir.as_deref() == Some(relative) {
            self.files.data_dir = Some(s);
        } else if self.bundle.files.encoder.as_deref() == Some(relative) {
            self.files.encoder = Some(s);
        } else if self.bundle.files.decoder.as_deref() == Some(relative) {
            self.files.decoder = Some(s);
        } else if self.bundle.files.vad_model.as_deref() == Some(relative) {
            self.files.vad_model = Some(s);
        }
    }
}

impl AssetBundle {
    /// Every relative path this bundle declares.
    pub fn declared_files(&self) -> Vec<String> {
        let mut out = Vec::new();
        for f in [
            &self.files.model,
            &self.files.voices,
            &self.files.tokens,
            &self.files.data_dir,
            &self.files.encoder,
            &self.files.decoder,
            &self.files.vad_model,
        ]
        .into_iter()
        .flatten()
        {
            out.push(f.clone());
        }
        out
    }

    /// True when `relative` names a directory entry (currently only the TTS
    /// `data_dir`, e.g. the espeak-ng data directory) rather than a file.
    pub fn is_dir_entry(&self, relative: &str) -> bool {
        self.files.data_dir.as_deref() == Some(relative)
    }
}

/// Hex-encode a byte slice without pulling in an extra dependency.
pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Download, verify and unpack a bundle into the model cache.
///
/// Enabled with the `neural` feature because it needs an HTTP client and
/// archive decoders. Verification itself (SHA-256 over a local file) is also
/// exposed so it can be tested without any network access.
#[cfg(feature = "neural")]
pub mod install {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Read;

    impl SpeechAssets {
        /// Install every bundle declared in the manifest, skipping ones that
        /// are already present and intact.
        pub fn install_all(&self, log: &mut dyn FnMut(String)) -> Result<(), AssetError> {
            for b in &self.manifest.bundles {
                self.install_bundle(&b.id, log)?;
            }
            Ok(())
        }

        /// Install a single bundle by id.
        pub fn install_bundle(
            &self,
            id: &str,
            log: &mut dyn FnMut(String),
        ) -> Result<(), AssetError> {
            let bundle = self
                .manifest
                .bundle(id)
                .ok_or_else(|| AssetError::UnknownBundle(id.to_string()))?
                .clone();
            if self.is_installed(id) {
                log(format!("[ok] {id}: already installed"));
                return Ok(());
            }
            let downloads = self.cache_dir.join("downloads");
            std::fs::create_dir_all(&downloads)?;
            let filename = bundle
                .url
                .rsplit('/')
                .next()
                .unwrap_or(&bundle.id)
                .to_string();
            let archive_path = downloads.join(&filename);
            if !archive_path.is_file() {
                log(format!("[dl] {id} <- {}", bundle.url));
                self.download(&bundle.url, &archive_path)?;
            } else {
                log(format!("[dl] {id}: using cached {}", archive_path.display()));
            }
            let size = std::fs::metadata(&archive_path)?.len();
            if bundle.bytes != 0 && size != bundle.bytes {
                return Err(AssetError::Size {
                    id: id.to_string(),
                    expected: bundle.bytes,
                    found: size,
                });
            }
            self.verify_sha256(&archive_path, &bundle.sha256, id)?;

            let models = self.models_dir();
            std::fs::create_dir_all(&models)?;
            match bundle.format {
                ArchiveFormat::Raw => {
                    std::fs::copy(&archive_path, models.join(&filename))?;
                }
                ArchiveFormat::TarBz2 => {
                    let file = std::fs::File::open(&archive_path)?;
                    let decoder = bzip2::read::BzDecoder::new(file);
                    let mut archive = tar::Archive::new(decoder);
                    archive.unpack(&models)?;
                }
            }
            if !self.is_installed(id) {
                return Err(AssetError::Install(format!(
                    "bundle '{id}' was unpacked but its declared files are missing"
                )));
            }
            log(format!("[ok] {id}: installed to {}", models.display()));
            Ok(())
        }

        /// Verify the SHA-256 of a local file against the manifest value.
        pub fn verify_sha256(&self, path: &Path, expected: &str, id: &str) -> Result<(), AssetError> {
            let found = sha256_file(path)?;
            if found != expected {
                return Err(AssetError::Checksum {
                    id: id.to_string(),
                    expected: expected.to_string(),
                    found,
                });
            }
            Ok(())
        }

        fn download(&self, url: &str, dest: &Path) -> Result<(), AssetError> {
            let tmp = dest.with_extension("part");
            let response = ureq::get(url)
                .call()
                .map_err(|e| AssetError::Install(format!("download failed for {url}: {e}")))?;
            let mut reader = response.into_reader();
            let mut file = std::fs::File::create(&tmp)?;
            std::io::copy(&mut reader, &mut file)?;
            file.sync_all()?;
            std::fs::rename(&tmp, dest)?;
            Ok(())
        }
    }

    /// SHA-256 of a file, streamed so large models do not need to be buffered.
    pub fn sha256_file(path: &Path) -> Result<String, AssetError> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 1 << 16];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex_encode(&hasher.finalize()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sha256_matches_known_value() {
            // sha256("abc")
            let dir = std::env::temp_dir().join("irl-sha-test");
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("abc.txt");
            std::fs::write(&p, b"abc").unwrap();
            assert_eq!(
                sha256_file(&p).unwrap(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../../../assets/speech/manifest.json");

    #[test]
    fn manifest_parses_and_validates() {
        let m = SpeechManifest::parse(SAMPLE).expect("manifest parses");
        assert_eq!(m.version, 1);
        assert!(m.bundle("kokoro-en-v0_19").is_some());
        assert!(m.bundle("whisper-tiny-en").is_some());
        assert!(m.bundle("silero-vad").is_some());
        let tts = m.first_of_kind(AssetKind::Tts).expect("tts bundle");
        assert_eq!(tts.voices.len(), 11);
        assert_eq!(tts.voices[0].sid, 0);
        assert_eq!(tts.voices[10].sid, 10);
        // Every voice id is unique.
        let mut ids: Vec<_> = tts.voices.iter().map(|v| v.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 11);
    }

    #[test]
    fn invalid_checksum_is_rejected() {
        let bad = SAMPLE.replace(
            "912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
            "not-a-checksum",
        );
        assert!(SpeechManifest::parse(&bad).is_err());
    }

    #[test]
    fn unknown_bundle_is_a_clear_error() {
        let m = SpeechManifest::parse(SAMPLE).unwrap();
        let a = SpeechAssets::new("/tmp/nonexistent-cache", m);
        let err = a.require_bundle("does-not-exist").unwrap_err();
        assert!(matches!(err, AssetError::UnknownBundle(_)));
        assert!(err.to_string().contains("does-not-exist"));
    }

    #[test]
    fn missing_bundle_reports_expected_path() {
        let m = SpeechManifest::parse(SAMPLE).unwrap();
        let a = SpeechAssets::new("/tmp/definitely-missing-cache", m);
        let err = a.require_bundle("silero-vad").unwrap_err();
        match err {
            AssetError::MissingAsset { id, path } => {
                assert_eq!(id, "silero-vad");
                assert!(path.contains("silero_vad.onnx"));
            }
            other => panic!("expected MissingAsset, got {other:?}"),
        }
    }

    #[test]
    fn hex_encoding_is_lowercase_and_padded() {
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xff]), "000fff");
        assert_eq!(hex_encode(&[]), "");
    }
}
