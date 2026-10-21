//! # rpg_persistence
//!
//! Versioned snapshots (JSON) and a transactional SQLite event store.
//!
//! * Replay determinism: the canonical event log is stored verbatim, so a
//!   snapshot + log can reproduce outcomes without re-running inference.
//! * Schema migrations are explicit and idempotent.
//! * Raw audio is *not* stored by default.

use rpg_domain::events::{Event, EventLog};
use rpg_domain::ids::EventId;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::World;
use rpg_sim::{SimConfig, SimOp, SimTransient, Simulation};
use rusqlite::{params, Connection};
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
    let text = serde_json::to_string_pretty(save)?;
    std::fs::write(path, text)?;
    Ok(())
}

pub fn load_json(path: impl AsRef<Path>) -> Result<SaveFile> {
    let text = std::fs::read_to_string(path)?;
    let save: SaveFile = serde_json::from_str(&text)?;
    if save.version != SAVE_VERSION {
        return Err(PersistenceError::Version {
            found: save.version,
            expected: SAVE_VERSION,
        });
    }
    Ok(save)
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
// SQLite event store
// ---------------------------------------------------------------------------

/// A local, transactional event store. One embedded database per save slot.
pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Idempotent schema setup with explicit versioning.
    pub fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                tick INTEGER NOT NULL,
                sim_minute INTEGER NOT NULL,
                actor TEXT,
                kind_json TEXT NOT NULL,
                caused_by_json TEXT NOT NULL,
                observed_by_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER NOT NULL,
                world_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )?;
        self.conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('version', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![SAVE_VERSION.to_string()],
        )?;
        Ok(())
    }

    pub fn append_event(&mut self, event: &Event) -> Result<()> {
        let kind = serde_json::to_string(&event.kind)?;
        let caused = serde_json::to_string(&event.caused_by)?;
        let observed = serde_json::to_string(&event.observed_by)?;
        self.conn.execute(
            "INSERT OR REPLACE INTO events(id, tick, sim_minute, actor, kind_json, caused_by_json, observed_by_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                event.id.raw() as i64,
                event.tick as i64,
                event.sim_minute as i64,
                event.actor.as_ref().map(|a| a.as_str()),
                kind,
                caused,
                observed
            ],
        )?;
        Ok(())
    }

    pub fn append_events(&mut self, events: &EventLog) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO events(id, tick, sim_minute, actor, kind_json, caused_by_json, observed_by_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for e in events.all() {
                stmt.execute(params![
                    e.id.raw() as i64,
                    e.tick as i64,
                    e.sim_minute as i64,
                    e.actor.as_ref().map(|a| a.as_str()),
                    serde_json::to_string(&e.kind)?,
                    serde_json::to_string(&e.caused_by)?,
                    serde_json::to_string(&e.observed_by)?
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn event_count(&self) -> Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))?;
        Ok(n as u64)
    }

    pub fn max_event_id(&self) -> Result<Option<EventId>> {
        let v: Option<i64> = self
            .conn
            .query_row("SELECT MAX(id) FROM events", [], |r| r.get(0))?;
        Ok(v.map(|n| EventId(n as u64)))
    }

    /// Persist a world snapshot inside a transaction.
    pub fn save_snapshot(&mut self, world: &World) -> Result<()> {
        let json = serde_json::to_string(world)?;
        self.conn.execute(
            "INSERT INTO snapshots(tick, world_json) VALUES (?1, ?2)",
            params![world.tick as i64, json],
        )?;
        Ok(())
    }

    pub fn latest_snapshot(&self) -> Result<Option<World>> {
        let mut stmt = self
            .conn
            .prepare("SELECT world_json FROM snapshots ORDER BY id DESC LIMIT 1")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let json: String = row.get(0)?;
            Ok(Some(serde_json::from_str(&json)?))
        } else {
            Ok(None)
        }
    }

    pub fn schema_version(&self) -> Result<u32> {
        let v: String = self.conn.query_row(
            "SELECT value FROM schema_meta WHERE key = 'version'",
            [],
            |r| r.get(0),
        )?;
        Ok(v.parse().unwrap_or(0))
    }
}

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
}
