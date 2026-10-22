//! The transactional SQLite event store (native targets only).

use crate::{Result, SAVE_VERSION};
use rpg_domain::events::{Event, EventLog};
use rpg_domain::ids::EventId;
use rpg_domain::world::World;
use rusqlite::{params, Connection};
use std::path::Path;

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
