//! Behavior-engine Phase 1 acceptance tests on the real town through the
//! real simulation pathways (test-spec §2): the liar-vs-skeptic deliverable
//! without special-case scripts, the provenance audit (M2), and information
//! noninterference for unreachable secrets (T1.5).

use rpg_domain::action::{Action, InvestigateTarget};
use rpg_domain::behavior_types::AcquisitionChannel;
use rpg_domain::events::EventKind;
use rpg_domain::ids::*;
use rpg_domain::world::{Fact, FactCategory, Provenance, Sensitivity};
use rpg_eval::Session;
use std::collections::BTreeSet;

/// A world-false rumor fact the player claims to know — a lie from the
/// world's perspective (asserting what the world says is not so).
fn plant_false_belief(session: &mut Session) -> FactId {
    let fact = FactId::new("f_test_lie");
    session.sim.world.facts.insert(
        fact.clone(),
        Fact {
            id: fact.clone(),
            statement: "The temple bell was stolen by the courier.".into(),
            truth: false,
            sensitivity: Sensitivity::Restricted,
            category: FactCategory::Rumor,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
    let (tick, minute) = (session.sim.world.tick, session.sim.world.minute_of_day);
    rpg_social::record_belief(
        &mut session.sim.world,
        &mut session.sim.events,
        tick,
        minute,
        &ActorId::new(PLAYER),
        &fact,
        true,
        0.9,
        Provenance::Assumed,
    );
    fact
}

/// T1.6 (deliverable): the same lie from the same speaker fools a trusting
/// NPC while a skeptic doubts it and requests additional evidence. The two
/// recipients differ only in their relationship state — no actor-id branches
/// anywhere in the mechanics.
#[test]
fn liar_fools_gullible_while_skeptic_seeks_evidence() {
    let seeds = 32;
    let mut gullible_believed = 0;
    let mut skeptic_believed = 0;
    let mut skeptic_sought = 0;
    for seed in 0..seeds {
        let mut s = Session::new(seed);
        let lie = plant_false_belief(&mut s);

        // Gullible recipient: deep trust toward the player.
        s.teleport_player_to("grocer");
        s.sim
            .world
            .actor_mut(&ActorId::new("grocer"))
            .unwrap()
            .relationship_mut(&ActorId::new(PLAYER))
            .trust = 0.9;
        let out = s.player_do(Action::ShareClaim {
            target: ActorId::new("grocer"),
            fact: lie.clone(),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        });
        assert!(matches!(
            out,
            rpg_domain::action::ValidationOutcome::Accepted
        ));
        let g_believed = s.sim.world.believes(&ActorId::new("grocer"), &lie);
        if g_believed {
            gullible_believed += 1;
        }

        // Skeptic recipient: distrust and suspicion toward the player.
        s.teleport_player_to("guard_captain");
        {
            let rel = s
                .sim
                .world
                .actor_mut(&ActorId::new("guard_captain"))
                .unwrap()
                .relationship_mut(&ActorId::new(PLAYER));
            rel.trust = -0.9;
            rel.suspicion = 0.9;
        }
        let out = s.player_do(Action::ShareClaim {
            target: ActorId::new("guard_captain"),
            fact: lie.clone(),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        });
        assert!(matches!(
            out,
            rpg_domain::action::ValidationOutcome::Accepted
        ));
        let s_believed = s.sim.world.believes(&ActorId::new("guard_captain"), &lie);
        if s_believed {
            skeptic_believed += 1;
        }

        // The skeptic's response policy on the aggregate state: an
        // unbelieved, weakly-supported claim triggers a request for more
        // evidence.
        let agg = rpg_behavior::inspect(&s.sim.world, &ActorId::new("guard_captain"), &lie);
        if rpg_behavior::skeptic_seeks_evidence(&agg, s_believed) {
            skeptic_sought += 1;
        }

        // Provenance on the skeptic's ledger: the claim's chain is preserved.
        let entries = s
            .sim
            .world
            .epistemics
            .evidence_about(&ActorId::new("guard_captain"), &lie);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].source_actor_id.as_ref().map(|a| a.as_str()),
            Some(PLAYER)
        );
        assert_eq!(entries[0].acquisition_channel, AcquisitionChannel::Told);
        assert!(entries[0].independent_corrob_group.is_some());
    }
    let g_rate = gullible_believed as f32 / seeds as f32;
    let s_rate = skeptic_believed as f32 / seeds as f32;
    assert!(
        g_rate > 0.8,
        "a trusting NPC should almost always be fooled by the lie ({gullible_believed}/{seeds})"
    );
    assert!(
        s_rate < 0.5 && g_rate - s_rate > 0.3,
        "the skeptic should doubt the same lie ({skeptic_believed}/{seeds} vs {gullible_believed}/{seeds})"
    );
    assert_eq!(
        skeptic_sought,
        seeds - skeptic_believed,
        "every unbelieved lie makes the skeptic seek more evidence"
    );
}

/// M2 audit: 100% of claims carry provenance — every `ClaimShared` event has
/// a matching ledger entry with source + channel + registered chain, and
/// every `EvidenceRecorded` event references a registered chain.
#[test]
fn every_claim_and_evidence_write_is_provenance_typed() {
    let mut s = Session::new(2026);

    // Real claim traffic through the authoritative simulation:
    // 1. The player plants a belief and shares it to the grocer.
    let lie = plant_false_belief(&mut s);
    s.teleport_player_to("grocer");
    let _ = s.player_do(Action::ShareClaim {
        target: ActorId::new("grocer"),
        fact: lie.clone(),
        assertive: true,
        deceptive: false,
        retold_chain: None,
    });
    // 2. The dockworker (authored believer) shares an operational fact to the
    //    guard captain — an NPC-to-NPC claim through the same pathway.
    {
        // Co-locate the two NPCs (fixture-only direct state edit; this test
        // performs no replay, so no recorded-op invariant is at stake).
        let captain = s
            .sim
            .world
            .actor(&ActorId::new("guard_captain"))
            .map(|a| (a.pos, a.location.clone()))
            .unwrap();
        let d = s.sim.world.actor_mut(&ActorId::new("dockworker")).unwrap();
        d.pos = captain.0;
        d.location = captain.1;
    }
    let out = s.sim.apply(&rpg_domain::action::ActionProposal {
        actor: ActorId::new("dockworker"),
        action: Action::ShareClaim {
            target: ActorId::new("guard_captain"),
            fact: FactId::new("f_shipments_diverted"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
        source: rpg_domain::action::ProposalSource::NpcPolicy,
        rationale: "audit fixture".into(),
    });
    assert!(
        !matches!(out, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "the dockworker knows and may share the shipment fact: {out:?}"
    );

    // 3. An intentional lie (T1.9/I-5) flows through the same authoritative
    // pathway: the player (who authored-believes the planted fact) asserts
    // deceptively — the claimed polarity flips, the world truth is
    // untouched, and the ClaimShared event carries the deceptive marker for
    // this audit.
    let out = s.player_do(Action::ShareClaim {
        target: ActorId::new("grocer"),
        fact: lie.clone(),
        assertive: true,
        deceptive: true,
        retold_chain: None,
    });
    assert!(
        matches!(out, rpg_domain::action::ValidationOutcome::Accepted),
        "a knower may lie: {out:?}"
    );
    assert!(
        !s.sim.world.facts.get(&lie).unwrap().truth,
        "a lie is a proposition choice, never a truth mutation"
    );

    let mut claim_count = 0;
    let mut evidence_count = 0;
    let mut deceptive_claims = 0;
    let mut seen_chains: BTreeSet<u64> = BTreeSet::new();
    for e in s.sim.events.all() {
        match &e.kind {
            EventKind::ClaimShared {
                from,
                to,
                fact,
                deceptive,
                ..
            } => {
                if *deceptive {
                    deceptive_claims += 1;
                }
                claim_count += 1;
                let entries = s.sim.world.epistemics.evidence_about(to, fact);
                assert!(
                    entries.iter().any(|ev| {
                        ev.source_actor_id.as_ref() == Some(from)
                            && ev.acquisition_channel == AcquisitionChannel::Told
                    }),
                    "claim {} from {} to {} has no provenance-bound evidence item",
                    fact.as_str(),
                    from,
                    to
                );
                // F6 (strengthened audit): every delivered claim produced a
                // causally linked told-path evidence write.
                let claim_id = e.id;
                let linked = s.sim.events.all().iter().any(|ev| {
                    matches!(
                        &ev.kind,
                        EventKind::EvidenceRecorded {
                            observer,
                            fact: delivered,
                            channel: AcquisitionChannel::Told,
                            ..
                        } if observer == to
                            && delivered == fact
                            && ev.caused_by.contains(&claim_id)
                    )
                });
                assert!(
                    linked,
                    "claim {} from {} to {} produced no causally linked evidence",
                    fact.as_str(),
                    from,
                    to
                );
            }
            EventKind::EvidenceRecorded {
                observer,
                fact,
                chain,
                ..
            } => {
                evidence_count += 1;
                seen_chains.insert(chain.raw());
                assert!(
                    s.sim.world.epistemics.chain(*chain).is_some(),
                    "evidence for {} on {} references an unregistered chain",
                    fact.as_str(),
                    observer
                );
                assert!(
                    !s.sim
                        .world
                        .epistemics
                        .evidence_about(observer, fact)
                        .is_empty(),
                    "the typed event references a missing ledger entry"
                );
                // F6 (strengthened audit): every causal link on an evidence
                // write points at a real ClaimShared delivery event.
                for c in &e.caused_by {
                    assert!(
                        matches!(
                            s.sim.events.get(*c).map(|x| &x.kind),
                            Some(EventKind::ClaimShared { .. })
                        ),
                        "evidence for {} on {} causally references a non-ClaimShared event",
                        fact.as_str(),
                        observer
                    );
                }
            }
            _ => {}
        }
    }
    assert!(claim_count >= 3, "the fixture must produce claim traffic");
    assert!(
        evidence_count >= 4,
        "expected ledger writes for all deliveries"
    );
    assert!(
        deceptive_claims >= 1,
        "M2: intentional lies are part of the audited claim traffic and carry provenance"
    );
    for id in seen_chains {
        let chain = s
            .sim
            .world
            .epistemics
            .chain(rpg_domain::behavior_types::SourceChainId(id));
        assert!(chain.is_some());
        assert!(
            !chain.unwrap().path.is_empty(),
            "every chain records its path"
        );
    }
}

/// T1.5 (integration): an unreachable secret's truth can change without any
/// observable difference in the actors' behaviour under a fixed seed.
#[test]
fn unreachable_secret_does_not_change_observables() {
    let run = |flip: bool| -> (u64, Option<String>) {
        let mut s = Session::new(2026);
        if flip {
            if let Some(f) = s.sim.world.facts.get_mut(&FactId::new("f_leader_identity")) {
                f.truth = !f.truth;
            }
        }
        s.teleport_player_to("grocer");
        let reply = s.say(
            "grocer",
            "I need to see your ledger. The one with the dock payments.",
        );
        s.goto("warehouse");
        let _ = s.player_do(Action::Investigate {
            target: InvestigateTarget::Location(LocationId::new("warehouse")),
        });
        s.ticks(20);
        (s.checksum(), reply)
    };
    let (checksum_a, reply_a) = run(false);
    let (checksum_b, reply_b) = run(true);
    assert_eq!(
        reply_a, reply_b,
        "the NPC's observable reply must not change"
    );
    assert_eq!(
        checksum_a, checksum_b,
        "unreachable secret truth must not change the observable event stream"
    );
}
