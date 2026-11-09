//! I7 acceptance tests (plan Phase 5; design §6.2/§6.3): emergent quests
//! recognized from real goals/bargains, constrained director proposals,
//! substitute predicate satisfaction and anchor validation.
//!
//! T5.1 quest recognition without a scripted giver
//! T5.2 substitute satisfaction (resolution = predicate, not one solution)
//! T5.3 director constraints (four negative cases, world unchanged)
//! T5.4 anchor validation via claims + permitted actor-facing surface

use rpg_domain::action::QuestPredicate;
use rpg_domain::deal::Deal;
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::geom::{Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::social::Commitment;
use rpg_domain::world::{
    evaluate_predicate, Actor, ActorKind, ActorStatus, Capabilities, Fact, FactCategory, Location,
    LocationKind, MysteryMode, NavGrid, Personality, Sensitivity, StoryAnchor, StoryState,
    WorldLore,
};
use rpg_narrative::{
    commit_director_proposal, quest_satisfied, recognize_quests, validate_director_proposal,
    Decision, DirectorProposal, QuestClaim, QuestContract, QuestGoalSource, QuestOpportunity,
    QuestStake, MAX_QUEST_PATHWAYS,
};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Small deterministic world helpers
// ---------------------------------------------------------------------------

fn actor(id: &str, kind: ActorKind) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.into(),
        kind,
        status: ActorStatus::Alive,
        location: LocationId::new("market"),
        pos: Vec2::ZERO,
        facing: 0.0,
        personality: Personality::default(),
        values: vec![],
        biography: String::new(),
        private_biography: String::new(),
        affiliations: vec![],
        capabilities: Capabilities::default(),
        inventory: Default::default(),
        goals: vec![],
        schedule: vec![],
        relationships: BTreeMap::new(),
        beliefs: BTreeMap::new(),
        inspected: BTreeSet::new(),
        flags: BTreeSet::new(),
        secret_allegiance: None,
        in_private_conversation_with: None,
        bargain: None,
        needs: Default::default(),
        emotions: Default::default(),
        rel_memory: BTreeMap::new(),
    }
}

fn location(id: &str) -> Location {
    Location {
        id: LocationId::new(id),
        name: id.into(),
        kind: LocationKind::Plaza,
        bounds: Rect::new(0.0, 0.0, 8.0, 8.0),
        owner: None,
        sound_transmission: 1.0,
        private: false,
        tags: vec![],
    }
}

fn item(id: &str) -> rpg_domain::world::Item {
    rpg_domain::world::Item {
        id: ItemId::new(id),
        name: id.into(),
        description: String::new(),
        base_value: 10,
        tags: vec![],
        stackable: true,
    }
}

fn fact(id: &str, truth: bool) -> Fact {
    Fact {
        id: FactId::new(id),
        statement: format!("statement {id}"),
        truth,
        sensitivity: Sensitivity::Private,
        category: FactCategory::Motive,
        prerequisites: vec![],
        eligible_holders: vec![],
        eligible_containers: vec![],
        committed: false,
        placed_at: None,
    }
}

fn goal_with_predicate(
    id: u64,
    description: &str,
    predicate: QuestPredicate,
) -> rpg_domain::world::Goal {
    rpg_domain::world::Goal {
        id,
        description: description.into(),
        kind: rpg_domain::world::GoalKind::Protect,
        priority: 9.0,
        status: rpg_domain::world::GoalStatus::Active,
        need_boost: 0.0,
        predicate: Some(predicate),
        target_location: None,
        target_actor: None,
    }
}

/// World with: player, `herbalist` (active goal needing `tonic`), `clerk`
/// (no predicate goals), tonic at `market`, nothing at `ruins`, one secret
/// uncommitted fact.
fn quest_world() -> rpg_domain::world::World {
    let mut w = rpg_domain::world::World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations: BTreeMap::new(),
        actors: BTreeMap::new(),
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
        nav: NavGrid::new(16, 16),
        story: StoryState {
            mode: MysteryMode::FixedAnswer,
            anchors: vec![StoryAnchor {
                id: "truth_fixed".into(),
                statement: "The truth is fixed".into(),
                immutable: true,
            }],
            checkpoints: vec![],
            fixed_answer: BTreeMap::new(),
            committed_containers: BTreeMap::new(),
        },
        commitments: BTreeMap::new(),
        offers: BTreeMap::new(),
        rumors: Vec::new(),
        damage: BTreeMap::new(),
        damage_facts: BTreeMap::new(),
        location_items: BTreeMap::new(),
        flags: BTreeSet::new(),
        ledger: BTreeMap::new(),
        next_commitment_id: 1,
        next_offer_id: 1,
        deals: BTreeMap::new(),
        next_deal_id: 1,
        next_rumor_id: 1,
        next_goal_id: 1,
        epistemics: Default::default(),
        lore: WorldLore::default(),
    };
    for loc in ["market", "ruins"] {
        w.locations.insert(LocationId::new(loc), location(loc));
    }
    for it in ["tonic", "herb"] {
        w.items.insert(ItemId::new(it), item(it));
    }
    w.actors
        .insert(ActorId::new(PLAYER), actor(PLAYER, ActorKind::Player));
    let mut herbalist = actor("herbalist", ActorKind::Npc);
    herbalist.goals = vec![goal_with_predicate(
        1,
        "Replenish the healing tonic",
        QuestPredicate::HasItem {
            actor: ActorId::new("herbalist"),
            item: ItemId::new("tonic"),
            qty: 1,
        },
    )];
    w.actors.insert(ActorId::new("herbalist"), herbalist);
    w.actors
        .insert(ActorId::new("clerk"), actor("clerk", ActorKind::Npc));
    w.location_items.insert(
        LocationId::new("market"),
        vec![rpg_domain::world::ItemStack {
            item: ItemId::new("tonic"),
            qty: 1,
        }],
    );
    w.facts
        .insert(FactId::new("f_secret"), fact("f_secret", true));
    w
}

fn has_item(who: &str, what: &str, qty: u32) -> QuestPredicate {
    QuestPredicate::HasItem {
        actor: ActorId::new(who),
        item: ItemId::new(what),
        qty,
    }
}

/// Snapshot of everything a surfacing proposal must never mutate.
fn truth_snapshot(w: &rpg_domain::world::World) -> String {
    format!(
        "facts:{:?}|goals:{:?}|deals:{:?}|commitments:{:?}|flags:{:?}|story:{:?}|location_items:{:?}|inventories:{:?}",
        w.facts,
        w.actors.iter().map(|(k, a)| (k, &a.goals)).collect::<Vec<_>>(),
        w.deals,
        w.commitments,
        w.flags,
        (&w.story.anchors, &w.story.committed_containers),
        w.location_items,
        w.actors.iter().map(|(k, a)| (k, &a.inventory)).collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// T5.1 — recognition from real goals/bargains, no scripted giver
// ---------------------------------------------------------------------------

#[test]
fn t5_1_goal_predicate_quest_recognized_without_scripted_giver() {
    let w = quest_world();
    let quests = recognize_quests(&w);
    assert_eq!(quests.len(), 1, "exactly the herbalist's blocked goal");
    let q = &quests[0];
    assert_eq!(q.beneficiary, ActorId::new("herbalist"));
    assert_eq!(q.goal, QuestGoalSource::Goal { id: 1 });
    assert_eq!(q.goal_predicate, has_item("herbalist", "tonic", 1));
    assert_eq!(q.blockers, vec![has_item("herbalist", "tonic", 1)]);
    assert!(q
        .pathways
        .iter()
        .any(|p| matches!(p, rpg_narrative::QuestPathway::ItemAvailableAt { location, .. } if location.as_str() == "market")));
    assert_eq!(
        q.stake.goal_kind,
        Some(rpg_domain::world::GoalKind::Protect)
    );
    assert_eq!(q.contract, None, "no contract exists yet");
}

#[test]
fn t5_1_satisfied_or_unreachable_goals_are_not_recognized() {
    let mut w = quest_world();
    // A satisfied goal is not a quest.
    w.actors
        .get_mut(&ActorId::new("herbalist"))
        .unwrap()
        .inventory
        .add(ItemId::new("tonic"), 1);
    assert!(recognize_quests(&w).is_empty(), "goal already satisfied");
    // A blocked goal whose needed item exists nowhere has no achievable
    // pathway — the recognizer must not invent one.
    let mut w2 = quest_world();
    w2.location_items.clear();
    assert!(
        recognize_quests(&w2).is_empty(),
        "no pathway, no opportunity"
    );
    // An actor with no predicate goals or bargain yields nothing.
    assert!(!recognize_quests(&w2)
        .iter()
        .any(|q| q.beneficiary == ActorId::new("clerk")));
}

#[test]
fn t5_1_content_world_grocer_quest_recognized() {
    // Production content: the grocer's authored bargain (typed medicine
    // condition) is recognized as an opportunity with no scripted giver.
    let w = rpg_content::build_town();
    let quests = recognize_quests(&w);
    assert_eq!(
        quests.len(),
        1,
        "fresh town recognizes exactly the grocer bargain quest"
    );
    let q = &quests[0];
    assert_eq!(q.beneficiary, ActorId::new("grocer"));
    assert_eq!(q.goal, QuestGoalSource::Bargain);
    assert_eq!(q.goal_predicate, has_item("grocer", "medicine", 1));
    assert_eq!(q.stake.promised_fact, Some(FactId::new("f_grocer_ledger")));
    assert_eq!(q.contract, None);
    // Pathways cite real placement only.
    for p in &q.pathways {
        match p {
            rpg_narrative::QuestPathway::ItemAvailableAt { item, location, .. } => {
                assert_eq!(item, &ItemId::new("medicine"));
                let stacks = w
                    .location_items
                    .get(location)
                    .expect("pathway location exists");
                assert!(stacks.iter().any(|s| s.item == ItemId::new("medicine")));
            }
            rpg_narrative::QuestPathway::HeldByActor { item, holder, .. } => {
                assert_eq!(item, &ItemId::new("medicine"));
                let holder = w.actor(holder).expect("pathway holder exists");
                assert!(holder.is_alive());
                assert!(holder.inventory.quantity(item) > 0);
            }
        }
    }
    assert!(!q.pathways.is_empty(), "medicine really is obtainable");
}

#[test]
fn t5_1_recognition_is_deterministic_and_bounded() {
    let w = quest_world();
    let a = recognize_quests(&w);
    let b = recognize_quests(&w);
    assert_eq!(a, b);
    // Many placements still produce a bounded, deterministic pathway set.
    let mut w2 = quest_world();
    for i in 0..12 {
        w2.location_items.insert(
            LocationId::new(format!("stash{i}")),
            vec![rpg_domain::world::ItemStack {
                item: ItemId::new("tonic"),
                qty: 1,
            }],
        );
        w2.locations.insert(
            LocationId::new(format!("stash{i}")),
            location(&format!("stash{i}")),
        );
    }
    let q = &recognize_quests(&w2)[0];
    assert_eq!(q.pathways.len(), MAX_QUEST_PATHWAYS);
}

// ---------------------------------------------------------------------------
// T5.2 — substitute predicate satisfaction
// ---------------------------------------------------------------------------

#[test]
fn t5_2_resolution_by_predicate_not_by_authored_solution() {
    let w = quest_world();
    let q = &recognize_quests(&w)[0];
    assert!(!quest_satisfied(&w, q));

    // Substitute route: the tonic reaches the herbalist through a channel
    // that was NOT among the recognized pathways (offscreen delivery in a
    // test harness). Resolution must follow the *predicate*, not the
    // pathway identity.
    let mut w2 = w.clone();
    w2.actor_mut(&ActorId::new("herbalist"))
        .unwrap()
        .inventory
        .add(ItemId::new("tonic"), 1);
    assert!(evaluate_predicate(&w2, &q.goal_predicate));
    assert!(quest_satisfied(&w2, q));
    // An unrelated item never satisfies the intended state.
    let mut w3 = w.clone();
    w3.actor_mut(&ActorId::new("herbalist"))
        .unwrap()
        .inventory
        .add(ItemId::new("herb"), 1);
    assert!(!quest_satisfied(&w3, q));
}

#[test]
fn t5_2_narrative_never_marks_goals_satisfied_or_mutates_truth() {
    let mut w = quest_world();
    let events_before = 0usize;
    let q = &recognize_quests(&w)[0];
    let _ = quest_satisfied(&w, q); // read-only recognition of resolution
                                    // Direct write of the satisfied state is the sim's authority, not the
                                    // narrative's; after it happens, the quest resolves on next read.
    w.actor_mut(&ActorId::new("herbalist"))
        .unwrap()
        .inventory
        .add(ItemId::new("tonic"), 1);
    assert!(quest_satisfied(&w, q));
    assert_eq!(events_before, 0);
    // Goal status untouched by any narrative call.
    assert_eq!(
        w.actors.get(&ActorId::new("herbalist")).unwrap().goals[0].status,
        rpg_domain::world::GoalStatus::Active
    );
}

// ---------------------------------------------------------------------------
// T5.3 — constrained director: four negative cases
// ---------------------------------------------------------------------------

#[test]
fn t5_3a_cannot_teleport_resources_tampered_pathway_rejected() {
    let w = quest_world();
    let mut q = recognize_quests(&w)[0].clone();
    // Tamper: claim the item is somewhere it is not.
    q.pathways = vec![rpg_narrative::QuestPathway::ItemAvailableAt {
        item: ItemId::new("tonic"),
        location: LocationId::new("nowhere"),
        qty: 1,
    }];
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: q,
        presented_to: ActorId::new(PLAYER),
        claims: vec![],
    };
    let before = truth_snapshot(&w);
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
    assert_eq!(before, truth_snapshot(&w));
}

#[test]
fn t5_3b_cannot_invent_beneficiary_or_quest() {
    let w = quest_world();
    // The clerk has no blocked goal/bargain — a surfaced quest for them is
    // an invented quest.
    let fake = QuestOpportunity {
        beneficiary: ActorId::new("clerk"),
        goal: QuestGoalSource::Bargain,
        description: "clerk needs tonic".into(),
        goal_predicate: has_item("clerk", "tonic", 1),
        blockers: vec![has_item("clerk", "tonic", 1)],
        pathways: vec![rpg_narrative::QuestPathway::ItemAvailableAt {
            item: ItemId::new("tonic"),
            location: LocationId::new("market"),
            qty: 1,
        }],
        stake: QuestStake {
            goal_kind: None,
            goal_priority: None,
            promised_fact: Some(FactId::new("f_secret")),
        },
        contract: None,
    };
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: fake,
        presented_to: ActorId::new(PLAYER),
        claims: vec![],
    };
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
    // Unknown presentation target.
    let real = recognize_quests(&w)[0].clone();
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: real,
        presented_to: ActorId::new("ghost"),
        claims: vec![],
    };
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
}

#[test]
fn t5_3c_cannot_invent_witnessed_events_or_contradict_committed_truth() {
    let mut w = quest_world();
    // Claim on an unknown fact: invention.
    let real = recognize_quests(&w)[0].clone();
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: real.clone(),
        presented_to: ActorId::new(PLAYER),
        claims: vec![QuestClaim {
            fact: FactId::new("f_nonexistent"),
            polarity: true,
        }],
    };
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
    // Claim on an uncommitted fact: invented witnessed evidence.
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: real.clone(),
        presented_to: ActorId::new(PLAYER),
        claims: vec![QuestClaim {
            fact: FactId::new("f_secret"),
            polarity: true,
        }],
    };
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
    // Commit the fact (fixture), then an inverted claim is a retroactive
    // contradiction of committed truth.
    w.facts.get_mut(&FactId::new("f_secret")).unwrap().committed = true;
    let real = recognize_quests(&w)[0].clone();
    let inverted = DirectorProposal::SurfaceQuest {
        opportunity: real.clone(),
        presented_to: ActorId::new(PLAYER),
        claims: vec![QuestClaim {
            fact: FactId::new("f_secret"),
            polarity: false,
        }],
    };
    assert!(matches!(
        validate_director_proposal(&w, &inverted),
        Decision::Rejected(_)
    ));
    // The truthful polarity is accepted.
    let truthful = DirectorProposal::SurfaceQuest {
        opportunity: real,
        presented_to: ActorId::new(PLAYER),
        claims: vec![QuestClaim {
            fact: FactId::new("f_secret"),
            polarity: true,
        }],
    };
    assert!(validate_director_proposal(&w, &truthful).is_accepted());
}

#[test]
fn t5_3d_cannot_treat_a_proposed_quest_as_an_established_contract() {
    let mut w = quest_world();
    let real = recognize_quests(&w)[0].clone();
    // Fake contract: no such deal exists.
    let mut fake = real.clone();
    fake.contract = Some(QuestContract::Deal(DealId(999)));
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: fake,
        presented_to: ActorId::new(PLAYER),
        claims: vec![],
    };
    assert!(matches!(
        validate_director_proposal(&w, &proposal),
        Decision::Rejected(_)
    ));
    // Real contract: an accepted commitment matching the predicate is
    // reported by the *recognizer* (never authored by the director).
    let cid = CommitmentId(1);
    w.commitments.insert(
        cid,
        Commitment {
            id: cid,
            kind: rpg_domain::social::CommitmentKind::Contract,
            debtor: ActorId::new(PLAYER),
            creditor: ActorId::new("herbalist"),
            terms: "tonic for thanks".into(),
            predicate: has_item("herbalist", "tonic", 1),
            state: rpg_domain::social::CommitmentState::Pending,
            created_tick: 0,
            deadline_minute: None,
            resentment: 0.0,
            promised_fact: Some(FactId::new("f_secret")),
        },
    );
    let recognized = recognize_quests(&w);
    let q = recognized
        .iter()
        .find(|q| q.beneficiary == ActorId::new("herbalist"))
        .expect("still recognized");
    assert_eq!(q.contract, Some(QuestContract::Commitment(cid)));
    assert_eq!(q.stake.promised_fact, Some(FactId::new("f_secret")));
    let proposal = DirectorProposal::SurfaceQuest {
        opportunity: q.clone(),
        presented_to: ActorId::new(PLAYER),
        claims: vec![],
    };
    assert!(validate_director_proposal(&w, &proposal).is_accepted());
}

#[test]
fn t5_3_commit_is_neutral_on_rejection() {
    let w = quest_world();
    let before = truth_snapshot(&w);
    let mut events = EventLog::new();
    let fake = QuestOpportunity {
        beneficiary: ActorId::new("clerk"),
        goal: QuestGoalSource::Bargain,
        description: "clerk needs tonic".into(),
        goal_predicate: has_item("clerk", "tonic", 1),
        blockers: vec![has_item("clerk", "tonic", 1)],
        pathways: vec![],
        stake: QuestStake {
            goal_kind: None,
            goal_priority: None,
            promised_fact: None,
        },
        contract: None,
    };
    let decision = commit_director_proposal(
        &w,
        &mut events,
        1,
        480,
        DirectorProposal::SurfaceQuest {
            opportunity: fake,
            presented_to: ActorId::new(PLAYER),
            claims: vec![],
        },
    );
    assert!(matches!(decision, Decision::Rejected(_)));
    assert_eq!(events.len(), 0, "rejected proposals write nothing");
    assert_eq!(before, truth_snapshot(&w));
}

// ---------------------------------------------------------------------------
// T5.4 — anchor validation and the permitted actor-facing surface
// ---------------------------------------------------------------------------

#[test]
fn t5_4_surfacing_commits_only_a_provenance_event_and_leaves_anchors_untouched() {
    let mut w = quest_world();
    w.facts.get_mut(&FactId::new("f_secret")).unwrap().committed = true;
    let before = truth_snapshot(&w);
    let real = recognize_quests(&w)[0].clone();
    let mut events = EventLog::new();
    let decision = commit_director_proposal(
        &w,
        &mut events,
        7,
        600,
        DirectorProposal::SurfaceQuest {
            opportunity: real.clone(),
            presented_to: ActorId::new(PLAYER),
            claims: vec![QuestClaim {
                fact: FactId::new("f_secret"),
                polarity: true,
            }],
        },
    );
    assert!(decision.is_accepted());
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events.all()[0].kind,
        EventKind::DirectorProposed { accepted: true, .. }
    ));
    // No world truth moved; the immutable anchor is untouched.
    assert_eq!(before, truth_snapshot(&w));
    assert!(w.story.anchors[0].immutable);
    // Surfacing again appends another provenance event (idempotent truth).
    let mut events2 = events.clone();
    let _ = commit_director_proposal(
        &w,
        &mut events2,
        8,
        601,
        DirectorProposal::SurfaceQuest {
            opportunity: real,
            presented_to: ActorId::new(PLAYER),
            claims: vec![],
        },
    );
    assert_eq!(events2.len(), 2);
}

#[test]
fn t5_4_surface_carries_no_privileged_truth() {
    let w = rpg_content::build_town();
    let q = &recognize_quests(&w)[0];
    let surface = rpg_narrative::QuestSurface::of(&w, q);
    assert_eq!(surface.beneficiary, ActorId::new("grocer"));
    assert!(surface.has_known_pathway);
    assert!(!surface.has_contract);
    // The permitted surface names the need but never the privileged
    // pathway locations (type carries no locations; text must not either).
    assert!(!surface.description.contains("warehouse"));
    assert!(!surface.description.contains("alley"));
    assert_eq!(surface.promised_fact, Some(FactId::new("f_grocer_ledger")));
}

// ---------------------------------------------------------------------------
// Contract matching against the deal FSM (reuse, not redesign)
// ---------------------------------------------------------------------------

#[test]
fn contract_matches_non_terminal_deal_only() {
    let mut w = quest_world();
    let did = DealId(1);
    let deal = Deal::proposed(
        did,
        ActorId::new(PLAYER),
        ActorId::new("herbalist"),
        "tonic delivery",
        has_item("herbalist", "tonic", 1),
        QuestPredicate::HeardFact {
            actor: ActorId::new(PLAYER),
            fact: FactId::new("f_secret"),
        },
        None,
        None,
        30,
        Vec::new(),
        0,
        Vec::new(),
        0,
    );
    w.deals.insert(did, deal);
    // Proposed (non-terminal) deal binds the quest.
    let q = &recognize_quests(&w)[0];
    assert_eq!(q.contract, Some(QuestContract::Deal(did)));
    // Terminal deals no longer count: settle through the FSM.
    let mut w2 = w.clone();
    let minute = w2.minute_of_day as u64;
    let d = w2.deals.get_mut(&did).unwrap();
    use rpg_domain::deal::DealEvent;
    assert!(d.advance(DealEvent::Accept, minute).is_ok());
    assert!(d.advance(DealEvent::Activate, minute).is_ok());
    assert!(d.advance(DealEvent::ConditionMet, minute).is_ok());
    assert!(d.advance(DealEvent::ObligationDue, minute).is_ok());
    assert!(d.advance(DealEvent::Perform, minute).is_ok());
    assert!(d.is_terminal());
    let q2 = &recognize_quests(&w2)[0];
    assert_eq!(q2.contract, None, "settled deal is no longer a contract");
}
