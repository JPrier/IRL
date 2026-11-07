//! Behavior-engine Phase 2 acceptance tests (test-spec T2.6, T2.7): the
//! hardcoded-grocer check scoped to the production new pathway, and the
//! rename-equivalence proof that no literal actor id gates a decision.
//!
//! T2.6 has two enforceable parts (plan §5 Phase 2; test-spec §3):
//!  * **(a)** a source audit over the new-pathway code only — every module of
//!    `crates/behavior` plus the new-pathway dispatch functions of
//!    `crates/agents` — for literal actor-id conditionals and grocer/fact
//!    literals. The legacy `heuristic_decision` fallback and the scenario
//!    content fixtures (`f_grocer_*` in `rpg_content`) are explicitly exempt
//!    until I-11.
//!  * **(b)** a rename-equivalence run: the new pathway on a world whose ids
//!    are uniformly permuted (grocer → baker, structure preserved) must
//!    produce isomorphic decisions.

use rpg_agents::{NpcPolicy, NpcRequestContext, ScriptedEngine};
use rpg_domain::action::{Action, QuestPredicate};
use rpg_domain::ids::*;
use rpg_domain::social::{Commitment, CommitmentKind, CommitmentState};
use rpg_domain::world::World;

// ---------------------------------------------------------------------------
// T2.6a — scoped source audit
// ---------------------------------------------------------------------------

fn read(rel: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Extract the source span of a top-level `fn name(...) { ... }` by brace
/// counting (format-string braces are balanced, so counting stays correct).
fn function_span(src: &str, name: &str) -> String {
    let needle = format!("fn {name}(");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("fn {name} not found"));
    let mut depth = 0usize;
    let mut seen_brace = false;
    let mut end = src.len();
    for (i, ch) in src[start..].char_indices() {
        match ch {
            '{' => {
                depth += 1;
                seen_brace = true;
            }
            '}' => {
                depth -= 1;
                if seen_brace && depth == 0 {
                    end = start + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    src[start..end].to_string()
}

fn assert_no_id_literals(span: &str, region: &str) {
    assert!(
        !span.contains("as_str() =="),
        "{region}: literal actor-id conditional in the new pathway: {span}"
    );
    assert!(
        !span.contains("grocer"),
        "{region}: grocer literal in the new pathway: {span}"
    );
}

#[test]
fn t2_6a_behavior_crate_has_no_actor_id_conditionals_or_grocer_literals() {
    for file in [
        "epistemics.rs",
        "mechanics.rs",
        "adapters.rs",
        "intentions.rs",
        "utility.rs",
        "appraisal.rs",
    ] {
        let src = read(&format!("../../crates/behavior/src/{file}"));
        assert!(
            !src.contains("grocer"),
            "behavior/src/{file}: grocer literal (new pathway must be content-driven)"
        );
        assert!(
            !src.contains("as_str() =="),
            "behavior/src/{file}: literal actor-id conditional"
        );
    }
}

#[test]
fn t2_6a_agents_new_pathway_has_no_actor_id_conditionals_or_grocer_literals() {
    let src = read("../../crates/agents/src/lib.rs");
    for name in [
        "canonical_bargain",
        "owed_disclosure",
        "deal_response_options",
        "deal_response_action",
        "describe_condition",
        "bargain_note",
    ] {
        let span = function_span(&src, name);
        assert_no_id_literals(&span, &format!("agents::{name}"));
    }
}

// ---------------------------------------------------------------------------
// T2.6b — rename equivalence (the strongest check)
// ---------------------------------------------------------------------------

/// A world whose ids are uniformly permuted: every occurrence of `from` in the
/// serialized canonical state (actor ids, fact ids, location ids, holder
/// lists, statements, lore) becomes `to`. The structure is preserved exactly.
fn renamed_world(from: &str, to: &str) -> World {
    let w = rpg_content::build_town();
    let json = serde_json::to_string(&w).unwrap();
    let renamed = json.replace(from, to);
    serde_json::from_str(&renamed)
        .unwrap_or_else(|e| panic!("renamed world failed to deserialize: {e}"))
}

/// Run the new decision pathway (a scripted model; the canonical bargain
/// overrides its plain speak) and return the action the policy committed.
fn new_pathway_decision(world: &World, npc: &str, utterance: &str) -> Action {
    let ctx = NpcRequestContext {
        world,
        npc: &ActorId::new(npc),
        speaker: &ActorId::new(PLAYER),
        utterance,
        history: &[],
    };
    let mut engine =
        ScriptedEngine::new().with_default(r#"{"action":"speak","text":"The weather is fine."}"#);
    NpcPolicy.decide(&ctx, 1, &mut engine).proposal.action
}

#[test]
fn t2_6b_decisions_are_isomorphic_under_actor_id_permutation() {
    let utterance = "I need to see your ledger. The one with the dock payments.";
    let base = new_pathway_decision(&rpg_content::build_town(), "grocer", utterance);
    let renamed = new_pathway_decision(&renamed_world("grocer", "baker"), "baker", utterance);

    // The base decision is the authored conditional bargain (new pathway).
    let Action::MakeOffer {
        condition: base_condition,
        disclose: base_disclose,
        note: base_note,
        ..
    } = &base
    else {
        panic!("expected the bargain offer, got {base:?}");
    };
    let Action::MakeOffer {
        condition: renamed_condition,
        disclose: renamed_disclose,
        note: renamed_note,
        ..
    } = &renamed
    else {
        panic!("expected the renamed bargain offer, got {renamed:?}");
    };
    // Isomorphic: same shape, ids mapped by the permutation.
    assert_eq!(
        (base_condition, base_disclose.as_ref()),
        (
            &Some(QuestPredicate::HasItem {
                actor: ActorId::new("grocer"),
                item: ItemId::new("medicine"),
                qty: 1,
            }),
            Some(&FactId::new("f_grocer_ledger"))
        ),
        "base bargain is the authored content-declared one"
    );
    assert_eq!(
        (renamed_condition, renamed_disclose.as_ref()),
        (
            &Some(QuestPredicate::HasItem {
                actor: ActorId::new("baker"),
                item: ItemId::new("medicine"),
                qty: 1,
            }),
            Some(&FactId::new("f_baker_ledger"))
        ),
        "the renamed world yields the isomorphic bargain — no literal id gates it"
    );
    assert_eq!(base_note, renamed_note);
}

/// The due-obligation pathway is equally id-free: with a due disclosure deal
/// the NPC honours it, and under the permutation the honouring decision maps
/// isomorphically.
#[test]
fn t2_6b_due_deal_honouring_is_isomorphic_under_permutation() {
    let utterance = "Anything new?";
    let make_due = |world: &mut World| -> DealId {
        // A due disclosure deal the grocer (or its rename) owes the player.
        let debtor = world
            .actors
            .keys()
            .find(|id| id.as_str() == "grocer" || id.as_str() == "baker")
            .unwrap()
            .clone();
        let id = DealId(world.next_deal_id);
        world.next_deal_id += 1;
        let fact = if debtor.as_str() == "grocer" {
            FactId::new("f_grocer_ledger")
        } else {
            FactId::new("f_baker_ledger")
        };
        world.deals.insert(
            id,
            rpg_domain::deal::Deal::proposed(
                id,
                debtor.clone(),
                ActorId::new(PLAYER),
                "bargain",
                QuestPredicate::HasItem {
                    actor: debtor.clone(),
                    item: ItemId::new("medicine"),
                    qty: 1,
                },
                QuestPredicate::HeardFact {
                    actor: ActorId::new(PLAYER),
                    fact,
                },
                None,
                None,
                30,
                Vec::new(),
                0,
                Vec::new(),
                0,
            ),
        );
        // Advance to PerformanceDue with the condition satisfied.
        let d = world.deals.get_mut(&id).unwrap();
        let _ = d.advance(rpg_domain::deal::DealEvent::Accept, 0);
        let _ = d.advance(rpg_domain::deal::DealEvent::Activate, 0);
        let _ = d.advance(rpg_domain::deal::DealEvent::ConditionMet, 0);
        let _ = d.advance(rpg_domain::deal::DealEvent::ObligationDue, 0);
        id
    };
    let mut base_world = rpg_content::build_town();
    make_due(&mut base_world);
    let mut renamed = renamed_world("grocer", "baker");
    make_due(&mut renamed);

    let base = new_pathway_decision(&base_world, "grocer", utterance);
    let renamed_action = new_pathway_decision(&renamed, "baker", utterance);
    let Action::ShareClaim {
        fact: base_fact, ..
    } = &base
    else {
        panic!("expected the due disclosure to be honoured, got {base:?}");
    };
    let Action::ShareClaim {
        fact: renamed_fact, ..
    } = &renamed_action
    else {
        panic!("expected the renamed due disclosure, got {renamed_action:?}");
    };
    assert_eq!(base_fact.as_str(), "f_grocer_ledger");
    assert_eq!(renamed_fact.as_str(), "f_baker_ledger");
}

// ---------------------------------------------------------------------------
// T2.7 — grocer scenario regression (unchanged scenario)
// ---------------------------------------------------------------------------

#[test]
fn t2_7_grocer_conditional_bargain_scenario_still_passes() {
    let r = rpg_eval::run_named("grocer_conditional_bargain", 2026).expect("scenario exists");
    assert!(
        r.passed(),
        "grocer_conditional_bargain must keep passing: {:?}",
        r.failures()
    );
}

/// The *new* pathway generates the same bargain from the authored content
/// (goal + resource obstacle + offer): with a scripted model that only
/// speaks, the policy still commits the authored conditional offer.
#[test]
fn t2_7_bargain_emerges_from_authored_content_in_the_new_pathway() {
    let action = new_pathway_decision(
        &rpg_content::build_town(),
        "grocer",
        "I need to see your ledger. The one with the dock payments.",
    );
    let Action::MakeOffer {
        condition,
        disclose,
        note,
        ..
    } = &action
    else {
        panic!("expected the authored bargain offer, got {action:?}");
    };
    assert_eq!(
        condition.as_ref(),
        Some(&QuestPredicate::HasItem {
            actor: ActorId::new("grocer"),
            item: ItemId::new("medicine"),
            qty: 1,
        })
    );
    assert_eq!(disclose.as_ref(), Some(&FactId::new("f_grocer_ledger")));
    assert_eq!(
        note.as_str(),
        "Bring me medicine and I will tell you what I know."
    );
}

// ---------------------------------------------------------------------------
// F4 — legacy-commitment fall-through to the due-deal pathway
// ---------------------------------------------------------------------------

/// A satisfied legacy commitment *without* a promised fact owes nothing more
/// from the commitment itself, and must not suppress the typed-deal pathway:
/// a due deal owed to the same speaker is still honoured (the old code
/// short-circuited on the legacy branch and returned no disclosure at all).
#[test]
fn owed_disclosure_legacy_commitment_falls_through_to_due_deal() {
    let mut world = rpg_content::build_town();
    let debtor = world
        .actors
        .keys()
        .find(|id| id.as_str() == "grocer")
        .unwrap()
        .clone();
    // (1) A satisfied legacy commitment without a promised fact.
    let commitment_id = CommitmentId(world.next_commitment_id);
    world.next_commitment_id += 1;
    world.commitments.insert(
        commitment_id,
        Commitment {
            id: commitment_id,
            kind: CommitmentKind::Favour,
            debtor: debtor.clone(),
            creditor: ActorId::new(PLAYER),
            terms: "old favour".into(),
            predicate: QuestPredicate::Flag("settled".into()),
            state: CommitmentState::Satisfied,
            created_tick: 0,
            deadline_minute: None,
            resentment: 0.0,
            promised_fact: None,
        },
    );
    // (2) A due typed deal whose benefit is the disclosure.
    let id = DealId(world.next_deal_id);
    world.next_deal_id += 1;
    world.deals.insert(
        id,
        rpg_domain::deal::Deal::proposed(
            id,
            debtor.clone(),
            ActorId::new(PLAYER),
            "bargain",
            QuestPredicate::HasItem {
                actor: debtor.clone(),
                item: ItemId::new("medicine"),
                qty: 1,
            },
            QuestPredicate::HeardFact {
                actor: ActorId::new(PLAYER),
                fact: FactId::new("f_grocer_ledger"),
            },
            None,
            None,
            30,
            Vec::new(),
            0,
            Vec::new(),
            0,
        ),
    );
    let d = world.deals.get_mut(&id).unwrap();
    let _ = d.advance(rpg_domain::deal::DealEvent::Accept, 0);
    let _ = d.advance(rpg_domain::deal::DealEvent::Activate, 0);
    let _ = d.advance(rpg_domain::deal::DealEvent::ConditionMet, 0);
    let _ = d.advance(rpg_domain::deal::DealEvent::ObligationDue, 0);

    // An utterance that matches no believed fact: the legacy branch has
    // nothing to offer, so the due deal must still be honoured.
    let action = new_pathway_decision(&world, "grocer", "Anything new?");
    let Action::ShareClaim { fact, .. } = &action else {
        panic!("expected the due deal's disclosure, got {action:?}");
    };
    assert_eq!(
        fact.as_str(),
        "f_grocer_ledger",
        "the legacy commitment must not suppress the due-deal disclosure"
    );
}
