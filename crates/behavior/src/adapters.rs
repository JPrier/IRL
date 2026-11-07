//! Compatibility adapters: mapping the legacy `Provenance`-based belief API
//! (`social::record_belief` and its callers) onto the evidence ledger, so the
//! provenance of every belief write is preserved while existing call sites
//! keep compiling and behaving.
//!
//! These adapters are used *inside* authoritative transitions (sim dispatch
//! and the social shims). Calibration defaults are design choices (plan §9):
//! explicitly approximate, acceptance is behavioural.

use crate::epistemics::AssertionDraft;
use rpg_domain::behavior_types::{AcquisitionChannel, SourceChainId};
use rpg_domain::ids::*;
use rpg_domain::world::{Provenance, World};

/// Build an [`AssertionDraft`] from a legacy [`Provenance`], computing the
/// recipient's perceived reliability:
///
/// * `Witnessed` / `Read` / `Inferred` / `Assumed` — fixed channel defaults.
/// * `Told { by }` — the recipient's relationship to `by` modulates the
///   channel default (trust raises it, suspicion lowers it).
pub fn draft_from_provenance(
    world: &World,
    observer: &ActorId,
    fact: &FactId,
    polarity: bool,
    provenance: &Provenance,
) -> AssertionDraft {
    let channel = match provenance {
        Provenance::Witnessed { .. } => AcquisitionChannel::Witnessed,
        Provenance::Told { .. } => AcquisitionChannel::Told,
        Provenance::Read { .. } => AcquisitionChannel::Read,
        Provenance::Inferred { .. } => AcquisitionChannel::Inferred,
        Provenance::Assumed => AcquisitionChannel::Assumed,
    };
    let (speaker, origin_event) = match provenance {
        Provenance::Witnessed { event } => (None, Some(*event)),
        Provenance::Told { by, event } => (Some(by.clone()), *event),
        _ => (None, None),
    };
    let reliability = match provenance {
        Provenance::Told { by, .. } => {
            let rel = world
                .actor(observer)
                .map(|a| a.relationship(by))
                .unwrap_or_default();
            let trust_01 = (1.0 + rel.trust) * 0.5;
            let suspicion_01 = (1.0 + rel.suspicion) * 0.5;
            (channel.default_reliability() + 0.25 * trust_01 - 0.25 * suspicion_01).clamp(0.1, 0.95)
        }
        _ => channel.default_reliability(),
    };
    AssertionDraft {
        speaker,
        proposition: fact.clone(),
        asserted_polarity: polarity,
        channel,
        origin_event,
        reliability,
        retold_chain: None,
        deceptive: false,
    }
}

/// Record an evidence item for a legacy provenance write, emitting the typed
/// `EvidenceRecorded` event. Used by the `social::record_belief` shim so all
/// canonical belief writes are event-sourced in the ledger too.
#[allow(clippy::too_many_arguments)]
pub fn record_provenance_evidence(
    world: &mut World,
    events: &mut rpg_domain::events::EventLog,
    tick: u64,
    minute: u32,
    observer: &ActorId,
    fact: &FactId,
    polarity: bool,
    provenance: &Provenance,
) -> SourceChainId {
    let draft = draft_from_provenance(world, observer, fact, polarity, provenance);
    let (evidence, chain) =
        crate::epistemics::record_evidence(&mut world.epistemics, observer, &draft, tick);
    events.push(
        tick,
        minute,
        Some(observer.clone()),
        rpg_domain::events::EventKind::EvidenceRecorded {
            observer: observer.clone(),
            fact: evidence.proposition_id.clone(),
            asserted_polarity: evidence.asserted_polarity,
            channel: evidence.acquisition_channel,
            source_actor: evidence.source_actor_id.clone(),
            chain,
            origin_event: evidence.origin_event_id,
        },
        Vec::new(),
        Vec::new(),
    );
    chain
}
