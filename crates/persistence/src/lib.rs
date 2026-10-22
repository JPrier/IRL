//! # rpg_persistence
//!
//! Versioned snapshots (JSON) and a transactional SQLite event store.
//!
//! * Replay determinism: the canonical event log is stored verbatim, so a
//!   snapshot + log can reproduce outcomes without re-running inference.
//! * Schema migrations are explicit and idempotent.
//! * Raw audio is *not* stored by default.

use rpg_domain::events::EventLog;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::World;
use rpg_sim::{SimConfig, SimOp, SimTransient, Simulation};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

pub const SAVE_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("unsupported save version {found} (expected {expected})")]
    Version { found: u32, expected: u32 },
    #[error("corrupt save: {0}")]
    Corrupt(String),
}

pub type Result<T> = std::result::Result<T, PersistenceError>;

/// A complete, versioned save: canonical world, event log, live RNG state,
/// transient movement/privacy state, and the proposal log needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveFile {
    pub version: u32,
    pub seed: u64,
    pub rng: DeterministicRng,
    pub world: World,
    pub events: EventLog,
    #[serde(default)]
    pub transient: SimTransient,
    #[serde(default)]
    pub ops: Vec<SimOp>,
    #[serde(default)]
    pub initial_world: Option<World>,
}

impl SaveFile {
    pub fn from_sim(sim: &Simulation) -> Self {
        Self {
            version: SAVE_VERSION,
            seed: sim.seed,
            rng: sim.rng.clone(),
            world: sim.world.clone(),
            events: sim.events.clone(),
            transient: sim.transient(),
            ops: sim.ops.clone(),
            initial_world: Some(sim.initial_world.clone()),
        }
    }
}

pub fn save_json(save: &SaveFile, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, to_json_string(save)?)?;
    Ok(())
}

pub fn load_json(path: impl AsRef<Path>) -> Result<SaveFile> {
    from_json_str(&std::fs::read_to_string(path)?)
}

/// Serialize a save to the canonical versioned JSON text. This is the one
/// save format on every platform: native files, browser storage and
/// exported/imported saves all carry exactly this text.
pub fn to_json_string(save: &SaveFile) -> Result<String> {
    Ok(serde_json::to_string_pretty(save)?)
}

/// Parse and version-check canonical save JSON.
pub fn from_json_str(text: &str) -> Result<SaveFile> {
    // Check the version before deserializing the whole world, so a save from
    // an incompatible build reports its version instead of a field error.
    let probe: serde_json::Value = serde_json::from_str(text)?;
    match probe.get("version").and_then(serde_json::Value::as_u64) {
        Some(v) if v == SAVE_VERSION as u64 => {}
        Some(v) => {
            return Err(PersistenceError::Version {
                found: v as u32,
                expected: SAVE_VERSION,
            })
        }
        None => return Err(PersistenceError::Corrupt("no save version".into())),
    }
    Ok(serde_json::from_value(probe)?)
}

/// Restore a `Simulation` from a save file, preserving live RNG and transient
/// movement state so subsequent steps match the original exactly.
pub fn restore(save: SaveFile, config: SimConfig) -> Simulation {
    let initial = save
        .initial_world
        .clone()
        .unwrap_or_else(|| save.world.clone());
    Simulation::restore_full(
        save.seed,
        save.world,
        save.events,
        save.rng,
        save.transient,
        save.ops,
        initial,
        config,
    )
}

/// Reconstruct canonical state from the recorded proposal/entry-point log.
pub fn replay(save: &SaveFile, config: SimConfig) -> Simulation {
    let initial = save
        .initial_world
        .clone()
        .unwrap_or_else(|| save.world.clone());
    Simulation::replay_ops(initial, save.seed, config, &save.ops)
}

// ---------------------------------------------------------------------------
// SQLite event store (native only)
// ---------------------------------------------------------------------------

// The browser build has no filesystem and must not link SQLite; it stores the
// same versioned `SaveFile` JSON through a platform adapter instead.
#[cfg(not(target_arch = "wasm32"))]
mod sqlite;
#[cfg(not(target_arch = "wasm32"))]
pub use sqlite::SqliteStore;

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_content::build_town;

    fn sim() -> Simulation {
        Simulation::new(build_town(), 42, SimConfig::default())
    }

    #[test]
    fn json_roundtrip_preserves_world_and_log() {
        let mut s = sim();
        let _ = s.step(vec![]);
        let save = SaveFile::from_sim(&s);
        let path = std::env::temp_dir().join(format!("rpg_save_{}.json", std::process::id()));
        save_json(&save, &path).unwrap();
        let loaded = load_json(&path).unwrap();
        assert_eq!(loaded.version, SAVE_VERSION);
        assert_eq!(loaded.events.checksum(), s.events.checksum());
        assert_eq!(loaded.world.actors.len(), s.world.actors.len());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sqlite_event_store_is_transactional_and_idempotent() {
        let mut s = sim();
        for _ in 0..5 {
            let _ = s.step(vec![]);
        }
        let mut store = SqliteStore::open_in_memory().unwrap();
        assert_eq!(store.schema_version().unwrap(), SAVE_VERSION);
        store.append_events(&s.events).unwrap();
        assert_eq!(store.event_count().unwrap(), s.events.len() as u64);
        // Re-appending is idempotent.
        store.append_events(&s.events).unwrap();
        assert_eq!(store.event_count().unwrap(), s.events.len() as u64);
        assert_eq!(
            store.max_event_id().unwrap(),
            s.events.all().last().map(|e| e.id)
        );
    }

    #[test]
    fn snapshot_roundtrip() {
        let s = sim();
        let mut store = SqliteStore::open_in_memory().unwrap();
        store.save_snapshot(&s.world).unwrap();
        let world = store.latest_snapshot().unwrap().unwrap();
        assert_eq!(world.actors.len(), s.world.actors.len());
        assert_eq!(world.minute_of_day, s.world.minute_of_day);
    }

    #[test]
    fn restore_preserves_event_checksum() {
        let mut s = sim();
        let _ = s.step(vec![]);
        let save = SaveFile::from_sim(&s);
        let restored = restore(save, SimConfig::default());
        assert_eq!(restored.events.checksum(), s.events.checksum());
    }

    #[test]
    fn save_load_continuation_matches_original_exactly() {
        use rpg_domain::action::{Action, ActionProposal, ProposalSource};
        use rpg_domain::ids::{ActorId, LocationId, PLAYER};
        let mut s = sim();
        let _ = s.apply(&ActionProposal {
            actor: ActorId::new(PLAYER),
            action: Action::MoveTo {
                location: LocationId::new("warehouse"),
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        });
        s.step(vec![]);
        let save = SaveFile::from_sim(&s);
        let mut restored = restore(save, SimConfig::default());
        for _ in 0..12 {
            s.step(vec![]);
            restored.step(vec![]);
        }
        assert_eq!(
            serde_json::to_string(&s.world).unwrap(),
            serde_json::to_string(&restored.world).unwrap(),
            "restored simulation diverged (RNG/transient state lost)"
        );
        assert_eq!(s.events.checksum(), restored.events.checksum());
    }

    #[test]
    fn proposal_replay_reconstructs_world() {
        use rpg_domain::action::{Action, ActionProposal, Loudness, ProposalSource};
        use rpg_domain::ids::{ActorId, PLAYER};
        let mut s = sim();
        let _ = s.apply(&ActionProposal {
            actor: ActorId::new(PLAYER),
            action: Action::Speak {
                target: Some(ActorId::new("grocer")),
                text: "hello".into(),
                loudness: Loudness::Normal,
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        });
        s.step(vec![]);
        let _ = s.apply(&ActionProposal {
            actor: ActorId::new(PLAYER),
            action: Action::Threaten {
                target: ActorId::new("grocer"),
                demand: "ledger".into(),
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        });
        for _ in 0..5 {
            s.step(vec![]);
        }
        let save = SaveFile::from_sim(&s);
        let replayed = replay(&save, SimConfig::default());
        assert_eq!(
            serde_json::to_string(&s.world).unwrap(),
            serde_json::to_string(&replayed.world).unwrap(),
            "proposal replay did not reconstruct the world"
        );
        assert_eq!(s.events.checksum(), replayed.events.checksum());
    }

    #[test]
    fn json_text_roundtrip_is_version_checked() {
        let sim = sim();
        let text = to_json_string(&SaveFile::from_sim(&sim)).unwrap();
        let back = from_json_str(&text).unwrap();
        assert_eq!(back.events.checksum(), sim.events.checksum());

        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["version"] = serde_json::json!(SAVE_VERSION + 1);
        assert!(matches!(
            from_json_str(&v.to_string()),
            Err(PersistenceError::Version { .. })
        ));
        v.as_object_mut().unwrap().remove("version");
        assert!(matches!(
            from_json_str(&v.to_string()),
            Err(PersistenceError::Corrupt(_))
        ));
        assert!(matches!(
            from_json_str("not json"),
            Err(PersistenceError::Json(_))
        ));
    }
}
