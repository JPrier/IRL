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

/// The current save format version.
pub const SAVE_VERSION: u32 = 2;

/// The save format version used before private_biography was repurposed to
/// stance prose. Saves with this version may carry raw secrets in
/// `Actor::private_biography` and organization-revealing goals/lore; the
/// load path sanitizes all of them (`sanitize_legacy_privacy`).
pub const LEGACY_SAVE_VERSION: u32 = 1;

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
    let mut probe: serde_json::Value = serde_json::from_str(text)?;
    match probe.get("version").and_then(serde_json::Value::as_u64) {
        Some(v) if v == SAVE_VERSION as u64 => {}
        Some(v) if v == LEGACY_SAVE_VERSION as u64 => {
            // A legacy (v1) save still carries the pre-repurposing authored
            // content: literal secrets in `Actor::private_biography` and
            // organization-revealing goals/lore. We sanitize it structurally
            // before deserializing, so none of it can enter world state,
            // regardless of how it is consumed downstream.
            sanitize_legacy_privacy(&mut probe);
            probe["version"] = serde_json::json!(SAVE_VERSION);
        }
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

/// Migrates the raw JSON of a legacy save (v1) so privacy-critical authored
/// content is safe before it is ever deserialized:
///
/// 1. `Actor::private_biography` is cleared wholesale: pre-repurposing saves
///    carried literal secrets in that field, and no prompt may ever render
///    them.
/// 2. Authored prompt-rendered text that names the concealed organization is
///    scrubbed. The concealed organization is identified structurally — any
///    faction with `"public": false`, matched by its id and full name as
///    token phrases. Shared lore is rendered into every resident's prompt, so
///    sentences and landmark entries naming it are removed unconditionally.
///    Per-actor goals are removed only when the actor has no canonical
///    access to the organization fact (neither an eligible holder of a
///    non-public `Organization` fact nor a member of the concealed faction);
///    members keep their own goals, and everything unrelated is preserved.
///
/// This is a privacy migration, not content hydration: nothing is rewritten
/// into new prose — offending items are simply removed.
fn sanitize_legacy_privacy(save: &mut serde_json::Value) {
    if let Some(world) = save.get_mut("world") {
        sanitize_world_privacy(world);
    }
    if let Some(initial_world) = save.get_mut("initial_world") {
        if initial_world.is_object() {
            sanitize_world_privacy(initial_world);
        }
    }
}

fn sanitize_world_privacy(world: &mut serde_json::Value) {
    sanitize_world_biographies(world);
    // Without a concealed faction there is nothing to scrub: a custom world
    // with only public organizations is untouched beyond the biography rule.
    let Some(concealed) = concealed_terms(world) else {
        return;
    };
    let authorized = authorized_actors(world);
    if let Some(actors) = world.get_mut("actors").and_then(|a| a.as_object_mut()) {
        for (id, actor) in actors.iter_mut() {
            if authorized.contains(id.as_str()) {
                continue;
            }
            let Some(goals) = actor.get_mut("goals").and_then(|g| g.as_array_mut()) else {
                continue;
            };
            goals.retain(|g| {
                !g.get("description")
                    .and_then(serde_json::Value::as_str)
                    .map(|d| mentions(&tokenize(d), &concealed))
                    .unwrap_or(false)
            });
        }
    }
    if let Some(lore) = world.get_mut("lore").and_then(|l| l.as_object_mut()) {
        for (key, value) in lore.iter_mut() {
            if key == "landmarks" {
                let Some(entries) = value.as_array_mut() else {
                    continue;
                };
                entries.retain(|e| {
                    !e.as_str()
                        .map(|s| mentions(&tokenize(s), &concealed))
                        .unwrap_or(false)
                });
            } else if let Some(text) = value.as_str() {
                let scrubbed = scrub_sentences(text, &concealed);
                if scrubbed != text {
                    *value = serde_json::Value::String(scrubbed);
                }
            }
        }
    }
}

fn sanitize_world_biographies(world: &mut serde_json::Value) {
    if let Some(actors) = world.get_mut("actors").and_then(|a| a.as_object_mut()) {
        for actor in actors.values_mut() {
            if let Some(actor_obj) = actor.as_object_mut() {
                if let Some(bio) = actor_obj.get_mut("private_biography") {
                    if bio.as_str().map(|s| !s.is_empty()).unwrap_or(false) {
                        *bio = serde_json::Value::String(String::new());
                    }
                }
            }
        }
    }
}

/// Token phrases identifying the concealed organization(s) of a world: every
/// non-public faction's id and full name, split into lowercase alphanumeric
/// tokens. `None` when the world has none.
fn concealed_terms(world: &serde_json::Value) -> Option<Vec<Vec<String>>> {
    let mut terms: Vec<Vec<String>> = Vec::new();
    if let Some(factions) = world.get("factions").and_then(|f| f.as_object()) {
        for faction in factions.values() {
            let public = faction
                .get("public")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            if public {
                continue;
            }
            for field in ["id", "name"] {
                if let Some(text) = faction.get(field).and_then(serde_json::Value::as_str) {
                    let phrase = tokenize(text);
                    if !phrase.is_empty() {
                        terms.push(phrase);
                    }
                }
            }
        }
    }
    (!terms.is_empty()).then_some(terms)
}

/// Actors with canonical access to the concealed organization: members of a
/// non-public faction, plus eligible holders of a non-public `Organization`
/// fact (e.g. the Watch captain who is hunting it).
fn authorized_actors(world: &serde_json::Value) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    if let Some(factions) = world.get("factions").and_then(|f| f.as_object()) {
        for faction in factions.values() {
            let public = faction
                .get("public")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            if public {
                continue;
            }
            if let Some(members) = faction.get("members").and_then(|m| m.as_array()) {
                for m in members {
                    if let Some(s) = m.as_str() {
                        out.insert(s.to_string());
                    }
                }
            }
        }
    }
    if let Some(facts) = world.get("facts").and_then(|f| f.as_object()) {
        for fact in facts.values() {
            let organization =
                fact.get("category").and_then(serde_json::Value::as_str) == Some("Organization");
            let concealed =
                fact.get("sensitivity").and_then(serde_json::Value::as_str) != Some("Public");
            if organization && concealed {
                if let Some(holders) = fact.get("eligible_holders").and_then(|h| h.as_array()) {
                    for h in holders {
                        if let Some(s) = h.as_str() {
                            out.insert(s.to_string());
                        }
                    }
                }
            }
        }
    }
    out
}

/// Lowercase alphanumeric runs of `text`.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether `tokens` contains any term phrase as a contiguous run. Phrases
/// match as whole tokens, so "trade" in a faction name never matches
/// "trading post", and a single-token id like "mafia" matches only that word.
fn mentions(tokens: &[String], phrases: &[Vec<String>]) -> bool {
    phrases
        .iter()
        .any(|p| tokens.windows(p.len()).any(|window| window == p.as_slice()))
}

/// Drop the sentences of `text` that mention a concealed term, keeping the
/// rest (and their punctuation) so unrelated lore survives the migration.
fn scrub_sentences(text: &str, phrases: &[Vec<String>]) -> String {
    let mut out = String::new();
    for sentence in text.split_inclusive('.') {
        let clean = sentence.trim();
        if clean.is_empty() || mentions(&tokenize(clean), phrases) {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(clean);
    }
    out
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

    #[test]
    fn test_legacy_save_secret_sanitization() {
        use rpg_agents::render_npc_context;
        use rpg_domain::ids::{ActorId, PLAYER};

        let mut sim = sim();
        let secret = "My secret is the server password: 12345";
        let actor_id = ActorId::new("grocer");

        // 1. Inject a raw secret into the actor's biography.
        sim.world
            .actor_mut(&actor_id)
            .expect("actor exists")
            .private_biography = secret.to_string();

        // 2. Create a save file and downgrade it to the legacy version (v1) to
        // simulate an old save written before private_biography was repurposed.
        let save = SaveFile::from_sim(&sim);
        assert_eq!(save.version, SAVE_VERSION);
        let json = to_json_string(&save).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        v["version"] = serde_json::json!(LEGACY_SAVE_VERSION);
        let legacy_json = v.to_string();

        // 3. Load the legacy save. This must trigger the sanitization boundary.
        let loaded = from_json_str(&legacy_json).expect("legacy save must load");
        assert_eq!(
            loaded.version, SAVE_VERSION,
            "legacy version must be migrated to the current version"
        );

        // 4. Verify structural sanitization: private_biography must be empty.
        let loaded_actor = loaded
            .world
            .actor(&actor_id)
            .expect("actor must survive migration");
        assert_eq!(
            loaded_actor.private_biography, "",
            "raw secret must be structurally cleared on migration"
        );

        // 5. Verify the prompt does not contain the raw secret.
        let speaker = ActorId::new(PLAYER);
        let prompt = render_npc_context(&loaded.world, &actor_id, &speaker, "Hello", &[]);
        assert!(
            !prompt.contains(secret),
            "prompt must not contain the raw secret"
        );

        // 6. Verify safe characterization is retained.
        assert!(
            prompt.contains("Marta Voss runs Voss & Daughter's"),
            "safe public biography must be retained"
        );
        assert!(
            prompt.contains("Things you believe:"),
            "authorized disclosure section must remain intact"
        );

        // 7. Verify normal (v2) save round-trip is NOT affected.
        let normal_json = to_json_string(&SaveFile::from_sim(&sim)).unwrap();
        let normal_loaded = from_json_str(&normal_json).expect("normal save must load");
        assert_eq!(
            normal_loaded
                .world
                .actor(&actor_id)
                .expect("actor must survive")
                .private_biography,
            secret,
            "normal save must not be sanitized"
        );
    }

    /// A v1 save also carried the old authored content: goals and shared lore
    /// that name the concealed organization. The migration must scrub those
    /// where they would be told to someone without canonical access (goals of
    /// unauthorized actors, all shared lore), keep what authorized members
    /// may know, and preserve everything unrelated. This reproduces the
    /// representative legacy content (the pre-repurposing town) on the real
    /// world.
    #[test]
    fn test_legacy_save_organization_references_are_sanitized() {
        use rpg_agents::render_npc_context;
        use rpg_domain::ids::{ActorId, PLAYER};
        use rpg_domain::world::{Goal, GoalKind, GoalStatus};

        let mut sim = sim();
        // Representative legacy content (what build_town authored before
        // private_biography was repurposed to stance prose).
        let legacy_economy = "Grain, timber, and river trade. The merchants' guild \
             dominates local commerce. The mafia controls smuggling and \
             protection rackets."
            .to_string();
        let legacy_events = "A recent warehouse fire destroyed a shipment of grain. \
             A courier went missing last month. The mafia is rumored to be \
             expanding its operations."
            .to_string();
        let legacy_landmarks = vec![
            "The Market Plaza, with its old stone well and weekly market".to_string(),
            "The Docks, where river barges unload grain and timber".to_string(),
            "The Mafia Hall, a discreet building in the warehouse district".to_string(),
        ];
        let mafia_goal = |id: u64, text: &str| Goal {
            id,
            description: text.to_string(),
            kind: GoalKind::Safety,
            priority: 6.0,
            status: GoalStatus::Active,
            predicate: None,
            target_location: None,
            target_actor: None,
        };
        let tavern = ActorId::new("tavern_keeper");
        {
            let a = sim.world.actor_mut(&tavern).unwrap();
            // Unauthorized actor: lore-adjacent goal naming the organization…
            a.goals.push(mafia_goal(90, "Avoid trouble from the mafia"));
            // …alongside unrelated content that must survive.
            a.goals.push(mafia_goal(91, "Tend the orchard"));
        }
        let head = ActorId::new("merchant_head");
        {
            let a = sim.world.actor_mut(&head).unwrap();
            // A member of the concealed faction: her goals are canonical
            // knowledge for her and must survive.
            a.goals.push(mafia_goal(92, "Expand mafia operations"));
        }
        sim.world.lore.economy = legacy_economy.clone();
        sim.world.lore.recent_events = legacy_events.clone();
        sim.world.lore.landmarks = legacy_landmarks.clone();
        let history_before = sim.world.lore.history.clone();

        // Downgrade to v1 and load.
        let json = to_json_string(&SaveFile::from_sim(&sim)).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        v["version"] = serde_json::json!(LEGACY_SAVE_VERSION);
        let loaded = from_json_str(&v.to_string()).expect("legacy save must load");

        // Unauthorized actor: the organization-naming goal is gone; the
        // unrelated goal survives.
        let tavern_loaded = loaded.world.actor(&tavern).unwrap();
        let descriptions: Vec<&str> = tavern_loaded
            .goals
            .iter()
            .map(|g| g.description.as_str())
            .collect();
        assert!(
            !descriptions.iter().any(|d| d.contains("mafia")),
            "unauthorized goal naming the concealed organization must be dropped: {descriptions:?}"
        );
        assert!(
            descriptions.contains(&"Tend the orchard"),
            "unrelated goals must survive: {descriptions:?}"
        );

        // Authorized member: her own goals are canonical knowledge for her.
        let head_loaded = loaded.world.actor(&head).unwrap();
        assert!(
            head_loaded
                .goals
                .iter()
                .any(|g| g.description == "Expand mafia operations"),
            "a concealed-faction member keeps her own goals"
        );

        // Shared lore: offending sentences/landmarks are removed, the rest
        // survives (privacy migration, not content deletion).
        assert_eq!(
            loaded.world.lore.economy,
            "Grain, timber, and river trade. The merchants' guild dominates \
             local commerce."
        );
        assert_eq!(
            loaded.world.lore.recent_events,
            "A recent warehouse fire destroyed a shipment of grain. A courier \
             went missing last month."
        );
        assert_eq!(
            loaded.world.lore.landmarks,
            vec![
                "The Market Plaza, with its old stone well and weekly market",
                "The Docks, where river barges unload grain and timber",
            ]
        );
        assert_eq!(loaded.world.lore.history, history_before);

        // The prompts of actors without canonical access no longer name it.
        let speaker = ActorId::new(PLAYER);
        let tavern_prompt = render_npc_context(&loaded.world, &tavern, &speaker, "Hello", &[]);
        assert!(
            !tavern_prompt.to_lowercase().contains("mafia"),
            "tavern keeper's prompt must not name the concealed organization"
        );

        // A normal (v2) save round-trips untouched.
        let tavern_goals_before = sim.world.actor(&tavern).unwrap().goals.len();
        let normal_loaded = from_json_str(&to_json_string(&SaveFile::from_sim(&sim)).unwrap())
            .expect("normal save must load");
        assert_eq!(normal_loaded.world.lore.economy, legacy_economy);
        assert_eq!(
            normal_loaded.world.actor(&tavern).unwrap().goals.len(),
            tavern_goals_before,
            "normal saves are never sanitized"
        );
        assert!(normal_loaded
            .world
            .actor(&tavern)
            .unwrap()
            .goals
            .iter()
            .any(|g| g.description.contains("mafia")));
    }

    /// A world with no concealed faction (a custom, public-only setup) is
    /// untouched by the organization scrub: no terms, nothing removed.
    #[test]
    fn test_legacy_save_without_concealed_faction_is_untouched() {
        use rpg_domain::ids::ActorId;
        use rpg_domain::world::{Goal, GoalKind, GoalStatus};

        let mut sim = sim();
        let mafia = sim
            .world
            .faction_mut(&rpg_domain::ids::FactionId::new("mafia"))
            .unwrap();
        mafia.public = true;
        let tavern = ActorId::new("tavern_keeper");
        sim.world.actor_mut(&tavern).unwrap().goals.push(Goal {
            id: 90,
            description: "Avoid trouble from the mafia".into(),
            kind: GoalKind::Safety,
            priority: 6.0,
            status: GoalStatus::Active,
            predicate: None,
            target_location: None,
            target_actor: None,
        });
        let economy_before = sim.world.lore.economy.clone();
        let tavern_goals_before = sim.world.actor(&tavern).unwrap().goals.len();

        let json = to_json_string(&SaveFile::from_sim(&sim)).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        v["version"] = serde_json::json!(LEGACY_SAVE_VERSION);
        let loaded = from_json_str(&v.to_string()).expect("legacy save must load");

        let tavern_loaded = loaded.world.actor(&tavern).unwrap();
        assert!(
            tavern_loaded.goals.len() == tavern_goals_before,
            "no concealed faction: goals untouched"
        );
        assert_eq!(loaded.world.lore.economy, economy_before);
    }
}
