//! Actor-local planner candidates (plan I-6, I6a slice; design §4.1 stage 4,
//! §4.3 medium path).
//!
//! Candidate generation is **grounded in what the actor itself holds**: its
//! own evidence ledger, its own relationships and emotions, its own needs and
//! custody, and real contract state it is a party to (or, for institutional
//! acts, real contract state an authority can see). Candidate generation
//! never reads fact truth values, other actors' private minds, or the
//! belief-formation path — the sim still validates every candidate like any
//! other proposal (adoption is never application; design §4.1 stage 6).
//!
//! Deliberately *not* every actor can do everything: a SANCTION candidate
//! exists only for an actor holding institutional authority *and* evidence
//! it actually holds; an EXCUSE candidate exists only for an authority
//! looking at an overdue due contract *and* holding citable evidence; a
//! QUERY candidate exists only when the actor's own ledger is conflicted or
//! thin on a proposition it is acting on. Actors without the grounding
//! generate no candidate at all (no indiscriminate sanctioning/excusing).

use rpg_behavior::{Feasibility, UtilityOption};
use rpg_domain::action::{Action, Loudness, QuestPredicate, SanctionKind};
use rpg_domain::deal::DealState;
use rpg_domain::emotions::{EmotionResponse, EmotionsConfig};
use rpg_domain::events::{Event, EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::needs::{NeedKind, NeedsConfig};
use rpg_domain::world::{Actor, ActorKind, FactCategory, World};
use std::collections::BTreeSet;

/// One planner candidate: the utility option the ranking kernel scores plus
/// the concrete typed action the plan would propose (validated by the sim).
#[derive(Clone, Debug)]
pub struct PlannerCandidate {
    pub option: UtilityOption,
    pub action: Action,
    /// Predicate that retires the adopted plan once it holds (`consider_replan`
    /// evaluates it; `None` = retired by the sim's outcome check for one-shot
    /// options).
    pub satisfaction: Option<QuestPredicate>,
    /// True when the plan is a single validated application (a question, a
    /// sanction, an excuse, an apology): the sim retires it once the
    /// corresponding outcome event exists.
    pub one_shot: bool,
    /// Trace description (never authoritative).
    pub summary: String,
}

/// The candidate option families (bounded set; the option id prefixes the
/// sim's outcome checks use).
pub mod option_ids {
    pub const QUERY: &str = "query";
    pub const SANCTION: &str = "sanction";
    pub const EXCUSE_DEAL: &str = "excuse_deal";
    pub const FLEE: &str = "flee";
    pub const REPORT_CRIME: &str = "report_crime";
    pub const SEEK_COMFORT_MOVE: &str = "seek_comfort_move";
    pub const SEEK_COMFORT_ASK: &str = "seek_comfort_ask";
    pub const RECONCILE: &str = "reconcile";
    /// Wealth-driven work (plan I-6 I6b §5.4): perform a repair the actor's
    /// own faction has funded an initiative for, and be paid a real wage.
    pub const WORK_REPAIR: &str = "work_repair";
    /// Epistemically bounded retelling (plan I-6 I6b §5.3): tell someone who
    /// does not yet know a proposition the actor believes and may disclose.
    pub const RETTELL: &str = "retell";
}

/// Cap on generated candidates — the per-activation work cap is enforced
/// here *and* by the caller (the sim truncates to its own budget; M6).
pub const MAX_CANDIDATES: usize = 16;

/// Whether an option id belongs to a one-shot family: the plan proposes one
/// validated application; the sim retires the plan once the corresponding
/// outcome event exists (never re-proposing a completed question/sanction/
/// excuse/apology/request).
pub fn is_one_shot_option(option: &str) -> bool {
    option.starts_with(option_ids::QUERY)
        || option.starts_with(option_ids::SANCTION)
        || option.starts_with(option_ids::EXCUSE_DEAL)
        || option.starts_with(option_ids::REPORT_CRIME)
        || option.starts_with(option_ids::SEEK_COMFORT_ASK)
        || option.starts_with(option_ids::RECONCILE)
        || option.starts_with(option_ids::RETTELL)
}

/// Generate the actor's local planner candidates. Deterministic; bounded —
/// `max_candidates` is the per-activation work cap, enforced *inside*
/// generation (an activation never considers more than its cap, whatever the
/// world throws at it; design §9.2.4, M6); no actor-id literals; no
/// world-truth reads. `already_sanctioned` is the formal punitive-sanction
/// ledger the actor must respect: an incident (evidence fact × subject)
/// that was already punished is never re-proposed (repeat punishment is a
/// policy violation, not a plan).
#[allow(clippy::too_many_arguments)]
pub fn planner_candidates(
    world: &World,
    npc: &ActorId,
    events: &EventLog,
    needs_config: &NeedsConfig,
    emotions_config: &EmotionsConfig,
    talk_radius: f32,
    max_candidates: usize,
    already_sanctioned: &BTreeSet<(FactId, ActorId)>,
) -> Vec<PlannerCandidate> {
    let Some(actor) = world.actor(npc) else {
        return Vec::new();
    };
    if !actor.is_alive() || actor.kind == ActorKind::Player {
        return Vec::new();
    }
    let cap = max_candidates.min(MAX_CANDIDATES);
    let mut out: Vec<PlannerCandidate> = Vec::new();
    epistemic_candidates(world, npc, &mut out, cap);
    emotion_candidates(
        world,
        npc,
        events,
        emotions_config,
        talk_radius,
        &mut out,
        cap,
    );
    institutional_candidates(world, npc, events, already_sanctioned, &mut out, cap);
    work_candidates(world, npc, needs_config, events, talk_radius, &mut out, cap);
    retell_candidates(world, npc, talk_radius, &mut out, cap);
    out
}

// ---------------------------------------------------------------------------
// Epistemic: conflicted propositions deserve a query to their own source
// ---------------------------------------------------------------------------

/// The propositions the actor *holds* — legacy beliefs plus its evidence
/// ledger (knowing-of is ledger-or-belief, matching `rpg_behavior::knows_of`).
/// Ledger-only knowledge (a received report without an accepted belief) is
/// still the actor's own knowledge for candidate grounding.
fn held_propositions(world: &World, npc: &ActorId, limit: usize) -> Vec<FactId> {
    let mut out: Vec<FactId> = Vec::new();
    if let Some(a) = world.actor(npc) {
        out.extend(a.beliefs.keys().cloned());
        for (fact, _) in world.epistemics.evidence.get(npc).into_iter().flatten() {
            out.push(fact.clone());
        }
    }
    out.sort();
    out.dedup();
    out.truncate(limit);
    out
}

/// A proposition the actor's own ledger reports as *contradicted* (both
/// polarities supported) is worth resolving: query the source actor of the
/// actor's most recent evidence item for it. Fully actor-local: the target
/// is taken from the actor's own provenance, never from who "really" knows.
fn epistemic_candidates(world: &World, npc: &ActorId, out: &mut Vec<PlannerCandidate>, cap: usize) {
    let facts = held_propositions(world, npc, 8);
    for fact in facts {
        if out.len() >= cap {
            return;
        }
        let agg = rpg_behavior::aggregate_about(&world.epistemics, npc, &fact);
        if !agg.contradicted {
            continue;
        }
        // The actor's own most recent evidence item with a source actor.
        let target = world
            .epistemics
            .evidence_about(npc, &fact)
            .iter()
            .rev()
            .find_map(|e| e.source_actor_id.clone());
        let Some(target) = target else {
            continue;
        };
        // The source must be present and near enough to actually ask.
        let near = matches!(
            (world.actor(&target), world.actor(npc)),
            (Some(t), Some(m)) if t.is_alive() && t.location == m.location && t.pos.distance(m.pos) <= 6.0
        );
        out.push(PlannerCandidate {
            option: UtilityOption {
                id: format!("{}:{fact}:{target}", option_ids::QUERY),
                goal_delta: 2.0,
                cost: 0.1,
                feasibility: if near {
                    Feasibility::Possible
                } else {
                    Feasibility::PhysicallyImpossible
                },
                // Resolving a conflicted report restores standing in one's
                // own knowledge (status) and calms the conflict.
                needs_gain: vec![(NeedKind::Status, 0.2)],
            },
            action: Action::Query {
                target,
                about: fact,
            },
            satisfaction: None,
            one_shot: true,
            summary: "resolve a conflicted report by asking its source".into(),
        });
        // One query per activation: bounded work.
        return;
    }
}

// ---------------------------------------------------------------------------
// Emotion gates: gated classes produce grounded response candidates
// ---------------------------------------------------------------------------

/// Consume the actor's gated emotion response classes (design §11 Phase 3;
/// the planner consumes what `Emotions::responses` gates). Each candidate is
/// grounded: the flee source is the actor's own feared edge, the comfort
/// target is the actor's own most-trusted edge, the reconcile target is the
/// actor's own most-salient exposure memory. Nothing is universal; nothing
/// fires without the actor's own state.
fn emotion_candidates(
    world: &World,
    npc: &ActorId,
    events: &EventLog,
    emotions_config: &EmotionsConfig,
    talk_radius: f32,
    out: &mut Vec<PlannerCandidate>,
    cap: usize,
) {
    let Some(actor) = world.actor(npc) else {
        return;
    };
    let gated: Vec<EmotionResponse> = actor.emotions.responses(emotions_config);
    if gated.is_empty() {
        return;
    }
    let alive_near: Vec<&Actor> = world
        .actors
        .values()
        .filter(|a| {
            a.is_alive()
                && a.id != *npc
                && a.location == actor.location
                && a.pos.distance(actor.pos) <= talk_radius
        })
        .collect();

    if out.len() >= cap {
        return;
    }
    if gated.contains(&EmotionResponse::Flee) {
        // The actor's own most-feared known actor is the thing to get away
        // from. The actor moves to a *known* place (the town map is public
        // physics; choosing a destination is not reading anyone's mind).
        let feared = actor
            .relationships
            .iter()
            .filter(|(id, _rel)| world.actor(id).is_some_and(|a| a.is_alive()))
            .max_by(|a, b| {
                a.1.fear
                    .partial_cmp(&b.1.fear)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|(_, rel)| rel.fear > 0.0)
            .map(|(id, _)| id.clone());
        if let Some(source) = feared {
            let source_pos = world.actor(&source).map(|a| a.pos);
            let dest = world
                .locations
                .values()
                .filter(|l| Some(l.center()) != source_pos)
                .max_by(|a, b| {
                    let da = source_pos.map(|p| p.distance(a.center())).unwrap_or(0.0);
                    let db = source_pos.map(|p| p.distance(b.center())).unwrap_or(0.0);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|l| l.id.clone());
            if let Some(dest) = dest {
                let arrived = actor.location == dest;
                out.push(PlannerCandidate {
                    option: UtilityOption {
                        id: format!("{}:{source}:{dest}", option_ids::FLEE),
                        goal_delta: 1.5,
                        cost: if arrived { 0.0 } else { 0.5 },
                        feasibility: Feasibility::Possible,
                        needs_gain: vec![(NeedKind::Safety, 0.4)],
                    },
                    action: Action::MoveTo {
                        location: dest.clone(),
                    },
                    // Retired once the actor stands somewhere else than the
                    // feared source's location.
                    satisfaction: Some(QuestPredicate::All(vec![
                        QuestPredicate::ActorAt {
                            actor: npc.clone(),
                            location: dest.clone(),
                        },
                        QuestPredicate::Not(Box::new(QuestPredicate::ActorAt {
                            actor: source.clone(),
                            location: dest,
                        })),
                    ])),
                    one_shot: false,
                    summary: format!("get away from {source} (afraid)"),
                });
            }
        }
    }

    if out.len() >= cap {
        return;
    }
    if gated.contains(&EmotionResponse::Retaliate) {
        // Retaliation is only available through a grounded institutional or
        // evidentiary route — anger alone commands nothing (no indiscriminate
        // retaliation): a witness routes through the nearest authority it
        // knows of; the authority's own sanction path is standing duty
        // (institutional_candidates), not an anger reaction.
        let maybe = believed_crime(world, npc, events);
        if let Some((fact, culprit)) = maybe {
            if let Some(authority) = nearest_authority(world, npc) {
                if out.len() >= cap {
                    return;
                }
                out.push(PlannerCandidate {
                    option: UtilityOption {
                        id: format!("{}:{culprit}:{fact}", option_ids::REPORT_CRIME),
                        goal_delta: 2.0,
                        cost: 0.2,
                        feasibility: Feasibility::Possible,
                        needs_gain: vec![(NeedKind::Safety, 0.2)],
                    },
                    action: Action::ReportCrime {
                        authority: authority.clone(),
                        evidence: vec![fact.clone()],
                    },
                    satisfaction: None,
                    one_shot: true,
                    summary: format!("report {culprit} to {authority} over {fact}"),
                });
            }
        }
    }

    if out.len() >= cap {
        return;
    }
    if gated.contains(&EmotionResponse::SeekComfort) || needs_company(world, npc) {
        // Seek the actor's own most-trusted nearby company: walk to them, or
        // — already together — ask for comfort (a REQUEST: asking is not
        // receiving).
        let trusted = alive_near
            .iter()
            .max_by(|a, b| {
                let ra = actor.relationship(&a.id);
                let rb = actor.relationship(&b.id);
                let sa = ra.trust_benevolence + ra.affection;
                let sb = rb.trust_benevolence + rb.affection;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id.clone());
        if let Some(target) = trusted {
            let near = alive_near.iter().any(|a| a.id == target);
            if near {
                out.push(PlannerCandidate {
                    option: UtilityOption {
                        id: format!("{}:{target}", option_ids::SEEK_COMFORT_ASK),
                        goal_delta: 1.0,
                        cost: 0.1,
                        feasibility: Feasibility::Possible,
                        needs_gain: vec![(NeedKind::Belonging, 0.2)],
                    },
                    action: Action::Speak {
                        target: Some(target.clone()),
                        text: "I could use some company right now.".into(),
                        loudness: Loudness::Normal,
                    },
                    satisfaction: None,
                    one_shot: true,
                    summary: format!("ask {target} for company"),
                });
            } else {
                let loc = world.actor(&target).map(|a| a.location.clone());
                if let Some(loc) = loc {
                    out.push(PlannerCandidate {
                        option: UtilityOption {
                            id: format!("{}:{target}", option_ids::SEEK_COMFORT_MOVE),
                            goal_delta: 1.0,
                            cost: 0.3,
                            feasibility: Feasibility::Possible,
                            needs_gain: vec![(NeedKind::Belonging, 0.3)],
                        },
                        action: Action::MoveTo {
                            location: loc.clone(),
                        },
                        satisfaction: Some(QuestPredicate::ActorAt {
                            actor: npc.clone(),
                            location: loc,
                        }),
                        one_shot: false,
                        summary: format!("seek company with {target}"),
                    });
                }
            }
        }
    }

    if out.len() >= cap {
        return;
    }
    if gated.contains(&EmotionResponse::Reconcile) {
        // The actor's own most-salient memory of being found out (or
        // wronging someone) names the party to make amends with — actor-local
        // memory, not an omniscient list of the wronged.
        let target = world
            .actor(npc)
            .and_then(|a| {
                a.rel_memory
                    .iter()
                    .flat_map(|(regarding, entries)| {
                        entries
                            .iter()
                            .filter(|m| {
                                m.interpretation.contains("deception")
                                    || m.interpretation.contains("insult")
                            })
                            .map(|m| (m.salience, regarding.clone(), m.tick))
                    })
                    .max_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.2.cmp(&b.2))
                    })
            })
            .map(|(_, regarding, _)| regarding);
        if let Some(target) = target {
            let near = matches!(
                (world.actor(&target), world.actor(npc)),
                (Some(t), Some(m)) if t.is_alive() && t.location == m.location && t.pos.distance(m.pos) <= talk_radius
            );
            out.push(PlannerCandidate {
                option: UtilityOption {
                    id: format!("{}:{target}", option_ids::RECONCILE),
                    goal_delta: 1.2,
                    cost: 0.2,
                    feasibility: if near {
                        Feasibility::Possible
                    } else {
                        Feasibility::PhysicallyImpossible
                    },
                    needs_gain: vec![(NeedKind::Status, 0.2)],
                },
                action: Action::Comfort {
                    target: target.clone(),
                },
                satisfaction: None,
                one_shot: true,
                summary: format!("make amends with {target}"),
            });
        }
    }
}

/// A belonging-need-deprived actor seeks company even without a grief spike
/// (design §6.4: satiation/thresholds; needs consume into action selection).
fn needs_company(world: &World, npc: &ActorId) -> bool {
    world
        .actor(npc)
        .map(|a| a.needs.value(NeedKind::Belonging) < 0.2)
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Institutional: authority-gated sanction and contract excuse
// ---------------------------------------------------------------------------

/// The believed crime fact (the actor's own ledger, not world truth) and the
/// actor it can ground a subject from: the origin event of the actor's own
/// strongest evidence — directly, or through the evidence's immutable source
/// chain when the report was retold to the actor (the chain preserves the
/// canonical origin the claim came from; retellings never rewrite it). A
/// fact whose provenance names no originator yields no subject — the
/// authority does not sanction on a bare assertion.
fn believed_crime(world: &World, npc: &ActorId, events: &EventLog) -> Option<(FactId, ActorId)> {
    believed_crimes(world, npc, events).into_iter().next()
}

/// Every believed crime the actor can ground (bounded by the held
/// proposition window), in the actor's deterministic held-proposition order.
/// Grounding per fact is the same chain-origin rule as [`believed_crime`];
/// the list shape lets standing duty skip an incident that was already
/// formally punished and act on the next one — an adjudicated incident must
/// never hide a genuinely distinct one behind it.
fn believed_crimes(world: &World, npc: &ActorId, events: &EventLog) -> Vec<(FactId, ActorId)> {
    let mut out: Vec<(FactId, ActorId)> = Vec::new();
    let Some(actor) = world.actor(npc) else {
        return out;
    };
    let facts = held_propositions(world, npc, 8);
    for fact in facts {
        let Some(f) = world.fact(&fact) else {
            continue;
        };
        if f.category != FactCategory::Crime {
            continue;
        }
        let agg = rpg_behavior::aggregate_about(&world.epistemics, npc, &fact);
        if agg.confidence < 0.6 {
            continue;
        }
        // Ground the subject in the actor's own evidence: the origin event of
        // its strongest evidence item (or of the chain that item carries) —
        // an event the actor holds provenance for, whose initiator is part
        // of that event's content.
        let strongest = world
            .epistemics
            .evidence_about(npc, &fact)
            .iter()
            .max_by(|a, b| {
                a.perceived_reliability
                    .partial_cmp(&b.perceived_reliability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
        let Some(evidence) = strongest else {
            continue;
        };
        let chain_origin = world
            .epistemics
            .chain(evidence.source_chain_id)
            .and_then(|c| c.origin_event);
        let origin = evidence.origin_event_id.or(chain_origin);
        let origin_actor = origin.and_then(|id| {
            events
                .all()
                .iter()
                .find(|e: &&Event| e.id == id)
                .and_then(|e| e.actor.clone())
        });
        let subject = origin_actor.or_else(|| match actor.beliefs.get(&fact).map(|b| &b.source) {
            Some(rpg_domain::world::Provenance::Witnessed { event }) => events
                .all()
                .iter()
                .find(|e| e.id == *event)
                .and_then(|e| e.actor.clone()),
            _ => None,
        });
        if let Some(subject) = subject {
            if subject != *npc && world.actor(&subject).is_some_and(|a| a.is_alive()) {
                out.push((fact, subject));
            }
        }
    }
    out
}

/// The nearest living authority actor (common knowledge of who the
/// institutional officers are), excluding the actor itself and players.
fn nearest_authority(world: &World, npc: &ActorId) -> Option<ActorId> {
    let me = world.actor(npc)?;
    world
        .actors
        .values()
        .filter(|a| {
            a.is_alive()
                && a.kind == ActorKind::Npc
                && a.id != *npc
                && (a.capabilities.can_arrest || a.has_flag("authority"))
        })
        .min_by_key(|a| a.pos.distance(me.pos).to_bits())
        .map(|a| a.id.clone())
}

/// Institutional candidates for an authority actor: (a) standing duty — a
/// grounded sanction on believed crime evidence (authority + evidence + a
/// chain-grounded subject; never a bare assertion, never an emotion
/// reaction); (b) excuse an *overdue due* contract with evidence the
/// authority actually holds. The gates are real contract state and the
/// authority's own knowledge — an authority without the grounding generates
/// nothing, and nobody else generates either. The standing duty also
/// respects the sanction identity policy: an (evidence fact × subject)
/// incident that was already formally punished is not re-proposed —
/// retold evidence about the same incident never produces repeat
/// punishment (the authoritative gate enforces the same identity).
fn institutional_candidates(
    world: &World,
    npc: &ActorId,
    events: &EventLog,
    already_sanctioned: &BTreeSet<(FactId, ActorId)>,
    out: &mut Vec<PlannerCandidate>,
    cap: usize,
) {
    let Some(actor) = world.actor(npc) else {
        return;
    };
    if !(actor.capabilities.can_arrest || actor.has_flag("authority")) || out.len() >= cap {
        return;
    }
    // Standing duty: the grounded sanction on real evidence. An incident
    // that was already formally punished is skipped — standing duty moves on
    // to the next believed crime instead of re-punishing.
    if let Some((fact, culprit)) = believed_crimes(world, npc, events)
        .into_iter()
        .find(|(fact, culprit)| !already_sanctioned.contains(&(fact.clone(), culprit.clone())))
    {
        let near = matches!(
            (world.actor(&culprit), world.actor(npc)),
            (Some(c), Some(m)) if c.location == m.location && c.pos.distance(m.pos) <= 6.0
        );
        if out.len() < cap {
            out.push(PlannerCandidate {
                option: UtilityOption {
                    id: format!("{}:{culprit}:{fact}", option_ids::SANCTION),
                    goal_delta: 2.5,
                    cost: 0.2,
                    feasibility: if near {
                        Feasibility::Possible
                    } else {
                        Feasibility::PhysicallyImpossible
                    },
                    needs_gain: vec![(NeedKind::Status, 0.2)],
                },
                action: Action::Sanction {
                    target: culprit.clone(),
                    kind: SanctionKind::Fine { amount: 10 },
                    evidence: vec![fact.clone()],
                },
                satisfaction: None,
                one_shot: true,
                summary: format!("fine {culprit} over {fact}"),
            });
        }
    }
    let now = world.day as u64 * 1440 + world.minute_of_day as u64;
    let citable: Option<FactId> = actor
        .beliefs
        .iter()
        .find(|(fid, b)| {
            b.believes_true
                && b.confidence >= 0.5
                && world
                    .fact(fid)
                    .is_some_and(|f| f.category != FactCategory::Rumor)
        })
        .map(|(fid, _)| fid.clone());
    let Some(evidence) = citable else {
        return;
    };
    let mut due: Vec<_> = world
        .deals
        .values()
        .filter(|d| {
            d.state == DealState::PerformanceDue
                && d.performance_deadline_minute
                    .is_some_and(|deadline| now >= deadline)
        })
        .collect();
    due.sort_by_key(|d| d.id.raw());
    for deal in due.into_iter().take(2) {
        if out.len() >= cap {
            return;
        }
        out.push(PlannerCandidate {
            option: UtilityOption {
                id: format!("{}:{}", option_ids::EXCUSE_DEAL, deal.id.raw()),
                goal_delta: 1.5,
                cost: 0.1,
                feasibility: Feasibility::Possible,
                needs_gain: vec![(NeedKind::Status, 0.1)],
            },
            action: Action::ExcuseDeal {
                deal: deal.id,
                evidence: vec![evidence.clone()],
            },
            satisfaction: None,
            one_shot: true,
            summary: format!(
                "excuse overdue deal #{} with authority evidence",
                deal.id.raw()
            ),
        });
    }
    let _ = events;
}

// ---------------------------------------------------------------------------
// Needs: deprivation routes into the same grounded candidates
// ---------------------------------------------------------------------------

/// Wealth-driven work (plan I-6 I6b, design §5.4 "production consumes
/// ingredients, labor and time"): a wealth-deprived actor who belongs to a
/// faction that has *funded a repair initiative* (announced through the
/// typed `FactionRepairInitiated` event — the faction's member-report group
/// decision, never the actor's truth access) and is standing at that
/// location may perform the real repair. The gain is only ever the real
/// validated wage the authoritative lowering pays for performed labor —
/// there is no artificial need boost anywhere in this family.
///
/// The plan is *not* one-shot: it is retired by its satisfaction predicate
/// (the location is actually repaired) or by the standard failure
/// abandonment (a misbelieved initiative fails validation honestly).
fn work_candidates(
    world: &World,
    npc: &ActorId,
    needs_config: &NeedsConfig,
    events: &EventLog,
    _talk_radius: f32,
    out: &mut Vec<PlannerCandidate>,
    cap: usize,
) {
    let _ = needs_config;
    let Some(actor) = world.actor(npc) else {
        return;
    };
    if !actor.needs.is_urgent(NeedKind::Wealth) {
        return;
    }
    // The funded initiatives the actor itself has observed (perception
    // boundary: an unobserved announcement is not the actor's knowledge).
    let mut initiatives: BTreeSet<(String, LocationId)> = BTreeSet::new();
    for e in events.all().iter().rev().take(64) {
        let observed = e.observed_by.is_empty() || e.observed_by.iter().any(|o| o == npc);
        if !observed {
            continue;
        }
        if let EventKind::FactionRepairInitiated {
            faction, location, ..
        } = &e.kind
        {
            initiatives.insert((faction.as_str().to_string(), location.clone()));
        }
    }
    for (faction_raw, location) in initiatives {
        if out.len() >= cap {
            return;
        }
        // Real institutional grounding: the actor belongs to the funding
        // faction (it performs the faction's coordinated plan) and stands at
        // the location (labor is performed where the work is).
        let member = world
            .faction(&FactionId::new(&faction_raw))
            .is_some_and(|f| f.members.contains(npc))
            || actor.affiliations.contains(&FactionId::new(&faction_raw));
        if !member || actor.location != location {
            continue;
        }
        out.push(PlannerCandidate {
            option: UtilityOption {
                id: format!(
                    "{}:{}:{}",
                    option_ids::WORK_REPAIR,
                    faction_raw,
                    location.as_str()
                ),
                goal_delta: 1.0,
                cost: 0.4,
                feasibility: Feasibility::Possible,
                needs_gain: vec![(NeedKind::Wealth, 0.25)],
            },
            action: Action::Repair {
                location: location.clone(),
            },
            satisfaction: Some(rpg_domain::action::QuestPredicate::LocationRepaired { location }),
            one_shot: false,
            summary: format!("work the repair initiative of {faction_raw} for a wage"),
        });
    }
}

/// The faction's believed state of a location it is responsible for, derived
/// **only from its members' own evidence ledgers** (design §5.5: group
/// beliefs are aggregates of communicated member reports — the faction has
/// no truth access). Public, pure, and used by the sim's faction tick; kept
/// here so the aggregation kernel lives beside the candidate grounding that
/// consumes its announcements.
pub fn faction_believed_damage(world: &World, faction: &FactionId, location: &LocationId) -> f32 {
    let Some(f) = world.faction(faction) else {
        return 0.0;
    };
    let mut believed = 0.0f32;
    for member in &f.members {
        let Some(fact) = world.damage_fact_of(location) else {
            continue;
        };
        let agg = rpg_behavior::aggregate_about(&world.epistemics, member, &fact);
        // A member contributes only what they themselves believe to be true.
        if world
            .actor(member)
            .and_then(|a| a.beliefs.get(&fact))
            .is_some_and(|b| b.believes_true)
        {
            believed += agg.confidence;
        }
    }
    believed.min(1.0)
}

// ---------------------------------------------------------------------------
// Retelling: epistemically bounded autonomous retransmission (§5.3)
// ---------------------------------------------------------------------------

/// Whether the actor may lawfully disclose this fact (the permission half of
/// the sim's authoritative disclosure validation — knowledge is not
/// permission, design §9.1.2). Mirrors the sim check as a *generation gate*
/// from fact metadata only; the sim still validates every candidate.
fn may_disclose(world: &World, actor: &ActorId, fact: &FactId) -> bool {
    let Some(f) = world.fact(fact) else {
        return false;
    };
    let Some(a) = world.actor(actor) else {
        return false;
    };
    use rpg_domain::world::Sensitivity;
    let holder = f.eligible_holders.is_empty() || f.eligible_holders.contains(actor);
    match f.sensitivity {
        Sensitivity::Public | Sensitivity::Restricted => true,
        Sensitivity::Private | Sensitivity::Secret => holder,
        Sensitivity::InnerCircle => a.has_flag("inner_circle") && holder,
    }
}

/// Epistemically bounded retelling (plan I-6 I6b, design §5.3 "each message
/// traverses a perceptual/social edge, carrying a source chain"): an actor
/// who *believes* a disclosable proposition may tell one nearby actor who
/// does not yet know it, **explicitly selecting the provenance chain the
/// retelling continues** (an actor holding several chains for one
/// proposition retells its intended chain — not necessarily its latest).
///
/// Grounding is actor-local only: the actor's own beliefs and ledger, fact
/// metadata (sensitivity/holders), and positional facts. The sim validates
/// the assertion (knowledge + permission) and the epistemic layer validates
/// the chain selection; a bogus selection falls back safely upstream. A
/// retelling is one-shot; hearing is neutral (the recipient records the
/// evidence with the carried chain — belief is still the recipient's roll).
fn retell_candidates(
    world: &World,
    npc: &ActorId,
    talk_radius: f32,
    out: &mut Vec<PlannerCandidate>,
    cap: usize,
) {
    const MAX_RETELLS: usize = 2;
    let Some(actor) = world.actor(npc) else {
        return;
    };
    let mut retells = 0usize;
    for (fact, belief) in actor.beliefs.iter() {
        if retells >= MAX_RETELLS || out.len() >= cap {
            return;
        }
        // Grounding: the actor must actually believe it (not merely know of
        // it) and be permitted to disclose it at all.
        if !(belief.believes_true && belief.confidence >= 0.5) || !may_disclose(world, npc, fact) {
            continue;
        }
        // The intended provenance chain: the actor's oldest held chain for
        // this proposition (explicit, validated selection — not the latest).
        let chain = world
            .epistemics
            .evidence_about(npc, fact)
            .first()
            .map(|e| e.source_chain_id);
        // One nearby listener who does not yet know the proposition at all
        // (rumour reach over real social edges — never a broadcast). The
        // §5.3 knowledge network traverses *ties*: a retelling needs a real
        // social edge (warmth), not just physical co-presence — neutral
        // strangers are not the actor's communication opportunity.
        let target = world
            .actors
            .values()
            .filter(|t| {
                t.id != *npc
                    && t.is_alive()
                    && t.location == actor.location
                    && t.pos.distance(actor.pos) <= talk_radius
                    && !rpg_behavior::knows_of(world, &t.id, fact)
                    && t.relationship(npc).warmth() >= 0.35
            })
            .map(|t| t.id.clone())
            .next();
        let Some(target) = target else {
            continue;
        };
        retells += 1;
        out.push(PlannerCandidate {
            option: UtilityOption {
                id: format!(
                    "{}:{}:{}",
                    option_ids::RETTELL,
                    fact.as_str(),
                    target.as_str()
                ),
                goal_delta: 0.5,
                cost: 0.05,
                feasibility: Feasibility::Possible,
                needs_gain: vec![(NeedKind::Belonging, 0.1)],
            },
            action: Action::ShareClaim {
                target,
                fact: fact.clone(),
                assertive: true,
                deceptive: false,
                retold_chain: chain,
            },
            satisfaction: None,
            one_shot: true,
            summary: format!(
                "retell {fact}{} (chain selected)",
                chain
                    .as_ref()
                    .map(|c| format!(" via chain {}", c.raw()))
                    .unwrap_or_default()
            ),
        });
    }
}

/// Whether the plan's one-shot outcome already exists (the sim retires
/// one-shot plans once their validated outcome event exists after the
/// adoption tick). Bounded to the last `WINDOW` events.
///
/// Matching is by the plan's **full parameters**, never by actor + event
/// kind alone (plan I-6 I6b hardening): a question about another fact, a
/// fine on another subject or a comfort for another target is a different
/// outcome and must not retire — or mis-attribute itself to — this plan.
/// The goal state the plan wanted is what matches (asked this question,
/// issued this sanction, …), whichever validated route produced it.
pub fn one_shot_completed(
    plan_action: &Action,
    actor: &ActorId,
    events: &EventLog,
    after_tick: u64,
) -> bool {
    const WINDOW: usize = 64;
    events.all().iter().rev().take(WINDOW).any(|e| {
        e.tick >= after_tick && e.actor.as_ref() == Some(actor) && outcome_of(plan_action, e)
    })
}

/// The typed event a one-shot plan's *own lowering* produced (the real
/// record — never invented): the question the asker asked, the sanction the
/// authority issued, the deal the authority excused, the report the witness
/// filed, the comfort the reconciler gave, the request the speaker spoke,
/// the claim the reteller told. Parameter-aware: the event must describe
/// the *same* goal state the plan's action describes.
fn outcome_of(plan_action: &Action, e: &Event) -> bool {
    match plan_action {
        Action::Query { target, about } => matches!(
            &e.kind,
            EventKind::QuestionAsked { from: _, to, about: a }
                if to == target && a == about
        ),
        Action::Sanction { target, kind, .. } => matches!(
            &e.kind,
            EventKind::SanctionIssued {
                subject,
                kind: k,
                ..
            } if subject == target && k.as_str() == kind.name()
        ),
        Action::ExcuseDeal { deal, .. } => {
            matches!(&e.kind, EventKind::DealExcused { deal: d, .. } if d == deal)
        }
        Action::ReportCrime { authority, .. } => matches!(
            &e.kind,
            EventKind::CrimeReported {
                authority: a,
                ..
            } if a == authority
        ),
        Action::Comfort { target } => matches!(
            &e.kind,
            EventKind::Comforted { from: _, to } if to == target
        ),
        Action::Speak { target, text, .. } => matches!(
            &e.kind,
            EventKind::SpeechEmitted {
                speaker: _,
                text: t,
                addressee,
                ..
            } if t == text && addressee.as_ref() == target.as_ref()
        ),
        Action::ShareClaim { target, fact, .. } => matches!(
            &e.kind,
            EventKind::ClaimShared {
                to,
                fact: f,
                ..
            } if to == target && f == fact
        ),
        _ => false,
    }
}
