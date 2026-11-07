//! Typed conditional contracts: the deal FSM (design §5.2; plan §4 is the
//! normative settlement specification this module implements).
//!
//! Naming: this module is deliberately `deal`, not `contract` —
//! `rpg_domain::contract` is the LLM action JSON schema and must not be
//! overloaded.
//!
//! Pinned semantics (plan §4):
//!
//! * FSM (§4.1): `Proposed -> Accepted -> Active -> ConditionMet ->
//!   PerformanceDue -> {Performed | Violated | Excused}`, plus
//!   `Rejected`/`Canceled` from the early states and a single
//!   `Renegotiated -> Active` cycle. Terminal transitions are exactly-once: a
//!   second terminal transition on the same id is rejected by validation.
//! * ConditionMet ≠ benefit delivered (§4.1): the trigger predicate
//!   (condition) and the promised benefit (performance) are separate typed
//!   predicates with separate transitions and separate events. A fulfillment
//!   *claim* never moves the FSM: only validated world predicates or settled
//!   `Effect::Transfer`/`Transform` evidence do.
//! * Escrow (§4.2): settlement is atomic inside one validated proposal — the
//!   sim validates the whole transfer bundle before mutating anything. There
//!   is **no** inventory lock/reservation mechanic in v1.
//! * Partial obligations (§4.3): installments settle each exactly-once and the
//!   deal performs only when all are settled; an un-split obligation delivered
//!   short stays `PerformanceDue` and emits a typed
//!   `DealPartialDelivery { got, expected }` event, mutating nothing
//!   contractual.
//! * Cancellation (§4.4): `Canceled` is legal from {Proposed, Accepted,
//!   Active} by mutual consent, the creditor, or the debtor **before
//!   ConditionMet and with no consideration exchanged** (give-on-acceptance
//!   consideration moves atomically at `Accepted`, §4.2). From
//!   `PerformanceDue`: creditor consent ⇒ `Canceled`,
//!   authority/institution evidence ⇒ `Excused`, a unilateral debtor ⇒ the
//!   `Violated` path. A debtor cancelling after ConditionMet is not
//!   `Canceled` — it is the `Violated` path. The pinned text gives the
//!   debtor *no* `Canceled` arrow from `Accepted`/`Active` once
//!   consideration has exchanged, and pins no alternative redirect for that
//!   case (the explicit `Violated` redirect is scoped to after
//!   ConditionMet), so the attempt is simply rejected — neutral, nothing
//!   mutates, and the creditor's options (consent, authority, deadline
//!   lapse) are unchanged.
//! * Deadlines (§4.5): the condition deadline applies to `Active` and the
//!   performance deadline to `PerformanceDue`, independently. Crossing emits
//!   exactly one typed `ObligationDue` event (idempotent) and starts the
//!   grace policy (renegotiation window), after which the deal is `Violated`
//!   with a typed event.
//!
//! [`transition`] is the pure, total, deterministic kernel the sim's
//! settlement paths call into; it is the direct Kani-harness target for
//! exactly-once terminal settlement and illegal-transition rejection (T6.3).

use crate::action::QuestPredicate;
use crate::ids::{ActorId, DealId, ItemId};
use crate::world::ItemStack;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// States and events
// ---------------------------------------------------------------------------

/// The pinned deal lifecycle states (plan §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DealState {
    Proposed,
    Accepted,
    Active,
    /// The trigger predicate holds — the obligation is *not* yet delivered.
    ConditionMet,
    /// The promised benefit is due; settlement happens from here.
    PerformanceDue,
    /// Terminal: the contracted benefit was delivered.
    Performed,
    /// Terminal: the deal failed (deadline lapsed, debtor reneged…).
    Violated,
    /// Terminal: released by authority/institution evidence.
    Excused,
    /// Terminal: the creditor declined the proposal.
    Rejected,
    /// Terminal: mutually / creditor / pre-consideration debtor cancellation.
    Canceled,
    /// The creditor consented to one renegotiation cycle.
    Renegotiated,
}

impl DealState {
    pub fn name(self) -> &'static str {
        match self {
            DealState::Proposed => "proposed",
            DealState::Accepted => "accepted",
            DealState::Active => "active",
            DealState::ConditionMet => "condition_met",
            DealState::PerformanceDue => "performance_due",
            DealState::Performed => "performed",
            DealState::Violated => "violated",
            DealState::Excused => "excused",
            DealState::Rejected => "rejected",
            DealState::Canceled => "canceled",
            DealState::Renegotiated => "renegotiated",
        }
    }

    /// Terminal states consume the deal: exactly-once (§4.1).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            DealState::Performed
                | DealState::Violated
                | DealState::Excused
                | DealState::Rejected
                | DealState::Canceled
        )
    }
}

/// Who requests a cancellation (§4.4 authorization matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Party {
    /// Both parties consent.
    Mutual,
    /// The creditor unilaterally.
    Creditor,
    /// The debtor unilaterally.
    Debtor,
}

/// Why a deal became `Violated` (recorded for appraisal/sanction follow-up).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationReason {
    /// The grace window after a crossed deadline elapsed.
    DeadlineLapsed,
    /// The debtor unilaterally walked away after the condition was met.
    DebtorReneged,
}

impl ViolationReason {
    pub fn name(self) -> &'static str {
        match self {
            ViolationReason::DeadlineLapsed => "deadline_lapsed",
            ViolationReason::DebtorReneged => "debtor_reneged",
        }
    }
}

/// A lifecycle event applied to a deal. The set of (state, event) pairs that
/// succeed is exactly the pinned §4.1 arrow set plus the §4.4 cancellation
/// matrix — nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DealEvent {
    Accept,
    Reject,
    Activate,
    ConditionMet,
    /// The obligation became due (ConditionMet → PerformanceDue).
    ObligationDue,
    Renegotiate,
    /// Leave the renegotiation cycle back into `Active`.
    Reactivate,
    /// The contracted benefit was delivered (validated predicate or settled
    /// transfer evidence).
    Perform,
    Violate {
        reason: ViolationReason,
    },
    /// Released by authority/institution evidence.
    Excuse,
    Cancel {
        by: Party,
    },
}

impl DealEvent {
    pub fn name(self) -> &'static str {
        match self {
            DealEvent::Accept => "accept",
            DealEvent::Reject => "reject",
            DealEvent::Activate => "activate",
            DealEvent::ConditionMet => "condition_met",
            DealEvent::ObligationDue => "obligation_due",
            DealEvent::Renegotiate => "renegotiate",
            DealEvent::Reactivate => "reactivate",
            DealEvent::Perform => "perform",
            DealEvent::Violate { reason } => match reason {
                ViolationReason::DeadlineLapsed => "violate:deadline_lapsed",
                ViolationReason::DebtorReneged => "violate:debtor_reneged",
            },
            DealEvent::Excuse => "excuse",
            DealEvent::Cancel { by } => match by {
                Party::Mutual => "cancel:mutual",
                Party::Creditor => "cancel:creditor",
                Party::Debtor => "cancel:debtor",
            },
        }
    }
}

/// Why a (state, event) pair was rejected by the transition kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DealTransitionError {
    /// The pinned FSM has no such arrow from `from`.
    Illegal {
        from: DealState,
        event: &'static str,
    },
    /// `from` is terminal and its terminal transition was already taken:
    /// exactly-once settlement/termination (§4.1, §4.6).
    Terminal {
        state: DealState,
        event: &'static str,
    },
}

impl std::fmt::Display for DealTransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DealTransitionError::Illegal { from, event } => {
                write!(f, "illegal deal transition {} from {}", event, from.name())
            }
            DealTransitionError::Terminal { state, event } => {
                write!(
                    f,
                    "deal already terminal ({}); {event} rejected (exactly-once)",
                    state.name()
                )
            }
        }
    }
}

/// The pure transition kernel (plan §4.1 + §4.4). Total, deterministic and
/// side-effect free: the same (state, event, facts) triple always yields the
/// same result. This is the production function every settlement path calls —
/// the later Kani harness (T6.3) proves exactly-once terminal settlement and
/// illegal-transition rejection over bounded deal/step spaces directly
/// against this symbol.
///
/// `consideration_exchanged` is the §4.4 cancellation fact: whether the
/// give-on-acceptance consideration has already moved (it moves atomically at
/// `Accepted`, §4.2 — see [`Deal::consideration_exchanged`]). In `Proposed`
/// nothing can have exchanged, so the flag is irrelevant there.
pub fn transition(
    state: DealState,
    event: DealEvent,
    consideration_exchanged: bool,
) -> Result<DealState, DealTransitionError> {
    use DealEvent as E;
    use DealState as S;
    if state.is_terminal() {
        return Err(DealTransitionError::Terminal {
            state,
            event: event.name(),
        });
    }
    let next = match (state, event) {
        // Proposal phase.
        (S::Proposed, E::Accept) => S::Accepted,
        (S::Proposed, E::Reject) => S::Rejected,
        (S::Proposed, E::Cancel { .. }) => S::Canceled,
        // Accepted, awaiting activation.
        (S::Accepted, E::Activate) => S::Active,
        (S::Accepted, E::Cancel { by }) => match by {
            // §4.4: the debtor may cancel as `Canceled` only with *no
            // consideration exchanged*. Once it has moved, there is no
            // pinned `Canceled` arrow for a unilateral debtor (see module
            // doc): the attempt is rejected, neutrally.
            Party::Debtor if consideration_exchanged => {
                return Err(DealTransitionError::Illegal {
                    from: state,
                    event: event.name(),
                })
            }
            _ => S::Canceled,
        },
        // Active: the condition may be met, or the deal cancelled early.
        (S::Active, E::ConditionMet) => S::ConditionMet,
        (S::Active, E::Cancel { by }) => match by {
            Party::Debtor if consideration_exchanged => {
                return Err(DealTransitionError::Illegal {
                    from: state,
                    event: event.name(),
                })
            }
            _ => S::Canceled,
        },
        // ConditionMet is a transient, non-cancellable state: the obligation
        // becomes due in the same validated step (design §7 step 5).
        (S::ConditionMet, E::ObligationDue) => S::PerformanceDue,
        // Performance due: settle, breach, excuse, or renegotiate once.
        (S::PerformanceDue, E::Perform) => S::Performed,
        (S::PerformanceDue, E::Violate { .. }) => S::Violated,
        (S::PerformanceDue, E::Excuse) => S::Excused,
        (S::PerformanceDue, E::Renegotiate) => S::Renegotiated,
        (S::PerformanceDue, E::Cancel { by }) => match by {
            // Creditor consent cancels (§4.4).
            Party::Mutual | Party::Creditor => S::Canceled,
            // A unilateral debtor past ConditionMet is the Violated path.
            Party::Debtor => S::Violated,
        },
        // Single renegotiation cycle.
        (S::Renegotiated, E::Reactivate) => S::Active,
        _ => {
            return Err(DealTransitionError::Illegal {
                from: state,
                event: event.name(),
            })
        }
    };
    Ok(next)
}

// ---------------------------------------------------------------------------
// Installments
// ---------------------------------------------------------------------------

/// One ordered installment of a split obligation (§4.3). Each installment
/// carries its own typed predicate/quantity and its own exactly-once
/// settlement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Installment {
    /// The typed predicate/quantity this installment settles.
    pub performance: QuestPredicate,
    /// The contracted quantity for quantity installments (0 = predicate-only).
    pub expected_qty: u32,
    /// Exactly-once settlement flag.
    pub settled: bool,
    pub settled_tick: Option<u64>,
}

/// Why an installment settlement was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallmentError {
    /// The installment was already settled: exactly-once per installment.
    AlreadySettled,
    IndexOutOfRange,
}

// ---------------------------------------------------------------------------
// Deal
// ---------------------------------------------------------------------------

/// Maximum renegotiation cycles (plan §4.1: a single renegotiation cycle).
pub const MAX_RENEGOTIATIONS: u32 = 1;

/// A typed conditional contract. Plain serializable canonical state; the
/// simulation owns the transitions, this module owns the rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Deal {
    pub id: DealId,
    pub debtor: ActorId,
    pub creditor: ActorId,
    pub terms: String,
    /// Trigger predicate: when it holds in the world, the obligation becomes
    /// due (ConditionMet). E.g. the grocer physically received the medicine.
    pub condition: QuestPredicate,
    /// Promised-benefit predicate: `PerformanceDue -> Performed` when it
    /// holds. E.g. the creditor actually heard the promised disclosure.
    pub performance: QuestPredicate,
    /// Condition deadline (absolute simulated minute) while `Active`.
    #[serde(default)]
    pub condition_deadline_minute: Option<u64>,
    /// Performance deadline (absolute simulated minute) while
    /// `PerformanceDue`.
    #[serde(default)]
    pub performance_deadline_minute: Option<u64>,
    /// Grace window (minutes) after a crossed deadline before `Violated`.
    #[serde(default = "default_grace")]
    pub grace_minutes: u64,
    pub state: DealState,
    /// Give-on-acceptance consideration: the debtor commits these resources;
    /// the sim validates custody at acceptance time (§4.2: no v1 lock) and
    /// moves them atomically at `Accepted`, before the FSM enters `Active`.
    #[serde(default)]
    pub give: Vec<ItemStack>,
    /// Money component of the give-on-acceptance consideration.
    #[serde(default)]
    pub money: i64,
    /// Split obligations (§4.3). Empty = un-split single obligation.
    #[serde(default)]
    pub installments: Vec<Installment>,
    /// Renegotiation cycles consumed (single-cycle policy).
    #[serde(default)]
    pub renegotiations: u32,
    pub created_tick: u64,
    #[serde(default)]
    pub condition_met_tick: Option<u64>,
    /// Idempotency: exactly one `ObligationDue` per crossed deadline (§4.5).
    #[serde(default)]
    pub condition_overdue_emitted: bool,
    #[serde(default)]
    pub performance_overdue_emitted: bool,
    /// Absolute minute until which the renegotiation/grace window runs.
    #[serde(default)]
    pub grace_until_minute: Option<u64>,
    #[serde(default)]
    pub resolved_minute: Option<u64>,
}

fn default_grace() -> u64 {
    30
}

/// How much of a delivery the contracted predicate expects, when it is a
/// quantity predicate: `(item, qty)` for `HasItem { actor, item, qty }`,
/// flattened through `All` combinators (first match wins; quantities are not
/// summed — a combined obligation is expressed with installments instead).
pub fn predicate_delivery(pred: &QuestPredicate) -> Option<(ActorId, ItemId, u32)> {
    match pred {
        QuestPredicate::HasItem { actor, item, qty } => Some((actor.clone(), item.clone(), *qty)),
        QuestPredicate::All(parts) => parts.iter().find_map(predicate_delivery),
        QuestPredicate::Any(parts) => parts.iter().find_map(predicate_delivery),
        _ => None,
    }
}

/// The item a quantity predicate delivers, when it is a `HasItem` predicate
/// (flattened through `All`/`Any`). Used to attribute a settlement bundle's
/// contents to the contracted quantity in `DealPartialDelivery`.
pub fn predicate_item(pred: &QuestPredicate) -> Option<ItemId> {
    match pred {
        QuestPredicate::HasItem { item, .. } => Some(item.clone()),
        QuestPredicate::All(parts) | QuestPredicate::Any(parts) => {
            parts.iter().find_map(predicate_item)
        }
        _ => None,
    }
}

impl Deal {
    /// A new deal in the `Proposed` state.
    #[allow(clippy::too_many_arguments)]
    pub fn proposed(
        id: DealId,
        debtor: ActorId,
        creditor: ActorId,
        terms: impl Into<String>,
        condition: QuestPredicate,
        performance: QuestPredicate,
        condition_deadline_minute: Option<u64>,
        performance_deadline_minute: Option<u64>,
        grace_minutes: u64,
        give: Vec<ItemStack>,
        money: i64,
        installments: Vec<Installment>,
        created_tick: u64,
    ) -> Self {
        Deal {
            id,
            debtor,
            creditor,
            terms: terms.into(),
            condition,
            performance,
            condition_deadline_minute,
            performance_deadline_minute,
            grace_minutes,
            state: DealState::Proposed,
            give,
            money,
            installments,
            renegotiations: 0,
            created_tick,
            condition_met_tick: None,
            condition_overdue_emitted: false,
            performance_overdue_emitted: false,
            grace_until_minute: None,
            resolved_minute: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Whether give-on-acceptance consideration has already moved (§4.2): it
    /// transfers atomically at `Accepted`, so any state past `Proposed` with
    /// a non-empty consideration (`give` items or `money`) has exchanged.
    /// §4.4 uses this to restrict the debtor's cancellation to the
    /// no-consideration case.
    pub fn consideration_exchanged(&self) -> bool {
        !matches!(self.state, DealState::Proposed) && (!self.give.is_empty() || self.money > 0)
    }

    /// Whether the single renegotiation cycle is still available.
    pub fn can_renegotiate(&self) -> bool {
        self.renegotiations < MAX_RENEGOTIATIONS
    }

    /// How many installments have settled (partial state is observable, §4.3).
    pub fn performed_installments(&self) -> usize {
        self.installments.iter().filter(|i| i.settled).count()
    }

    pub fn total_installments(&self) -> usize {
        self.installments.len()
    }

    /// Apply a lifecycle event through the pinned transition kernel,
    /// maintaining the bookkeeping fields. This is the guarded settlement
    /// entry point (§4.6): the state check happens here, so replaying the
    /// recorded, deterministic transition sequence cannot double-settle.
    pub fn advance(
        &mut self,
        event: DealEvent,
        minute: u64,
    ) -> Result<DealState, DealTransitionError> {
        let next = transition(self.state, event, self.consideration_exchanged())?;
        self.state = next;
        match next {
            DealState::ConditionMet => {
                if self.condition_met_tick.is_none() {
                    self.condition_met_tick = Some(minute);
                }
            }
            DealState::Renegotiated => self.renegotiations += 1,
            DealState::Proposed
            | DealState::Accepted
            | DealState::Active
            | DealState::PerformanceDue => {}
            _ => self.resolved_minute = Some(minute),
        }
        Ok(next)
    }

    /// Settle one installment exactly-once (§4.3). Returns whether *all*
    /// installments are now settled (the caller then performs the deal).
    pub fn settle_installment(
        &mut self,
        index: usize,
        tick: u64,
    ) -> Result<bool, InstallmentError> {
        let Some(inst) = self.installments.get_mut(index) else {
            return Err(InstallmentError::IndexOutOfRange);
        };
        if inst.settled {
            return Err(InstallmentError::AlreadySettled);
        }
        inst.settled = true;
        inst.settled_tick = Some(tick);
        Ok(self.installments.iter().all(|i| i.settled))
    }

    /// Mark the deadline for `state` as crossed: emits `ObligationDue` once
    /// (idempotent by state check, §4.5) and returns whether this call was
    /// the one that crossed it.
    pub fn cross_deadline(&mut self, performance: bool, minute: u64) -> bool {
        let already = if performance {
            self.performance_overdue_emitted
        } else {
            self.condition_overdue_emitted
        };
        if already {
            return false;
        }
        if performance {
            self.performance_overdue_emitted = true;
        } else {
            self.condition_overdue_emitted = true;
        }
        self.grace_until_minute = Some(minute + self.grace_minutes);
        true
    }

    /// Whether the grace window after a crossed deadline has elapsed.
    pub fn grace_elapsed(&self, minute: u64) -> bool {
        self.grace_until_minute.is_some_and(|until| minute > until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deal() -> Deal {
        Deal::proposed(
            DealId(1),
            ActorId::new("debtor"),
            ActorId::new("creditor"),
            "test",
            QuestPredicate::Flag("condition".into()),
            QuestPredicate::Flag("performance".into()),
            None,
            None,
            30,
            Vec::new(),
            0,
            Vec::new(),
            0,
        )
    }

    /// The pinned §4.1 happy path.
    #[test]
    fn pinned_lifecycle_is_exact() {
        let mut d = deal();
        let path = [
            (DealEvent::Accept, DealState::Accepted),
            (DealEvent::Activate, DealState::Active),
            (DealEvent::ConditionMet, DealState::ConditionMet),
            (DealEvent::ObligationDue, DealState::PerformanceDue),
            (DealEvent::Perform, DealState::Performed),
        ];
        for (event, want) in path {
            assert_eq!(d.advance(event, 1).unwrap(), want);
        }
        assert!(d.is_terminal());
    }

    /// Exactly-once: a second terminal transition on the same id is rejected.
    #[test]
    fn terminal_states_are_consumed_exactly_once() {
        let mut d = deal();
        d.advance(DealEvent::Reject, 1).unwrap();
        for event in [
            DealEvent::Perform,
            DealEvent::Violate {
                reason: ViolationReason::DeadlineLapsed,
            },
            DealEvent::Excuse,
            DealEvent::Reject,
            DealEvent::Accept,
        ] {
            assert!(matches!(
                d.advance(event, 2),
                Err(DealTransitionError::Terminal { .. })
            ));
        }
    }

    /// Illegal arrows are rejected (T2.2): the pinned diagram only.
    #[test]
    fn illegal_transitions_are_rejected() {
        // Skipping Accepted (Proposed → Active directly).
        assert!(matches!(
            transition(DealState::Proposed, DealEvent::Activate, false),
            Err(DealTransitionError::Illegal { .. })
        ));
        // Performing before the condition was met.
        assert!(matches!(
            transition(DealState::Active, DealEvent::Perform, false),
            Err(DealTransitionError::Illegal { .. })
        ));
        // Cancelling during the transient ConditionMet state.
        assert!(matches!(
            transition(
                DealState::ConditionMet,
                DealEvent::Cancel {
                    by: Party::Creditor
                },
                false
            ),
            Err(DealTransitionError::Illegal { .. })
        ));
        // Renegotiating twice in a row.
        assert!(matches!(
            transition(DealState::Renegotiated, DealEvent::Renegotiate, false),
            Err(DealTransitionError::Illegal { .. })
        ));
        // Excusing before anything is due.
        assert!(matches!(
            transition(DealState::Accepted, DealEvent::Excuse, false),
            Err(DealTransitionError::Illegal { .. })
        ));
    }

    /// §4.4 cancellation authorization matrix. The debtor's `Canceled`
    /// cancellation is legal only *before ConditionMet* *and* *with no
    /// consideration exchanged* (§4.2: consideration moves atomically at
    /// `Accepted`); the pinned text pins no redirect for the
    /// with-consideration early case, so the kernel has no such arrow.
    #[test]
    fn cancellation_authorization_matrix() {
        // Proposed: nothing can have exchanged; any party cancels.
        for by in [Party::Mutual, Party::Creditor, Party::Debtor] {
            assert_eq!(
                transition(DealState::Proposed, DealEvent::Cancel { by }, false).unwrap(),
                DealState::Canceled
            );
        }
        // Accepted/Active: mutual and creditor always cancel; the debtor
        // only while no consideration has exchanged.
        for state in [DealState::Accepted, DealState::Active] {
            for by in [Party::Mutual, Party::Creditor] {
                for exchanged in [false, true] {
                    assert_eq!(
                        transition(state, DealEvent::Cancel { by }, exchanged).unwrap(),
                        DealState::Canceled
                    );
                }
            }
            assert_eq!(
                transition(state, DealEvent::Cancel { by: Party::Debtor }, false).unwrap(),
                DealState::Canceled
            );
            assert!(matches!(
                transition(state, DealEvent::Cancel { by: Party::Debtor }, true),
                Err(DealTransitionError::Illegal { .. })
            ));
        }
        // PerformanceDue: creditor consent ⇒ Canceled; authority ⇒ Excused;
        // unilateral debtor ⇒ the Violated path (regardless of the
        // consideration fact).
        for exchanged in [false, true] {
            assert_eq!(
                transition(
                    DealState::PerformanceDue,
                    DealEvent::Cancel {
                        by: Party::Creditor
                    },
                    exchanged
                )
                .unwrap(),
                DealState::Canceled
            );
            assert_eq!(
                transition(
                    DealState::PerformanceDue,
                    DealEvent::Cancel { by: Party::Debtor },
                    exchanged
                )
                .unwrap(),
                DealState::Violated
            );
        }
        assert_eq!(
            transition(DealState::PerformanceDue, DealEvent::Excuse, false).unwrap(),
            DealState::Excused
        );
    }

    /// §4.4: `Deal::consideration_exchanged` tracks the §4.2 acceptance
    /// escrow — nothing before `Accept`, everything with a non-empty
    /// consideration after it, nothing when the terms are empty.
    #[test]
    fn consideration_exchanged_tracks_acceptance_escrow() {
        let mut d = deal(); // empty consideration
        assert!(!d.consideration_exchanged());
        d.advance(DealEvent::Accept, 1).unwrap();
        assert!(!d.consideration_exchanged(), "empty terms never exchange");
        let mut d = deal();
        d.give = vec![ItemStack {
            item: ItemId::new("medicine"),
            qty: 1,
        }];
        assert!(!d.consideration_exchanged(), "nothing moved before accept");
        d.advance(DealEvent::Accept, 1).unwrap();
        assert!(d.consideration_exchanged(), "items moved at Accepted");
        let mut d = deal();
        d.money = 5;
        d.advance(DealEvent::Accept, 1).unwrap();
        assert!(d.consideration_exchanged(), "money moved at Accepted");
    }

    /// ConditionMet ≠ Performed are distinct states (T2.3/T6.3).
    #[test]
    fn condition_met_is_not_performed() {
        let mut d = deal();
        d.advance(DealEvent::Accept, 1).unwrap();
        d.advance(DealEvent::Activate, 2).unwrap();
        d.advance(DealEvent::ConditionMet, 3).unwrap();
        assert_eq!(d.state, DealState::ConditionMet);
        assert!(!d.is_terminal());
        // The obligation still must become due before it can perform.
        assert!(matches!(
            transition(DealState::ConditionMet, DealEvent::Perform, false),
            Err(DealTransitionError::Illegal { .. })
        ));
        d.advance(DealEvent::ObligationDue, 4).unwrap();
        assert_eq!(
            d.advance(DealEvent::Perform, 5).unwrap(),
            DealState::Performed
        );
    }

    /// §4.5 idempotent deadline crossing.
    #[test]
    fn deadline_crossing_is_idempotent() {
        let mut d = deal();
        d.advance(DealEvent::Accept, 1).unwrap();
        d.advance(DealEvent::Activate, 2).unwrap();
        assert!(d.cross_deadline(false, 100));
        assert!(!d.cross_deadline(false, 101));
        assert!(!d.cross_deadline(false, 102));
        assert_eq!(d.grace_until_minute, Some(130));
    }

    /// §4.3 installments settle exactly-once and are observable.
    #[test]
    fn installments_settle_exactly_once() {
        let mut d = deal();
        d.installments = vec![
            Installment {
                performance: QuestPredicate::Flag("a".into()),
                expected_qty: 1,
                settled: false,
                settled_tick: None,
            },
            Installment {
                performance: QuestPredicate::Flag("b".into()),
                expected_qty: 2,
                settled: false,
                settled_tick: None,
            },
        ];
        assert!(!d.settle_installment(0, 5).unwrap());
        assert!(matches!(
            d.settle_installment(0, 6),
            Err(InstallmentError::AlreadySettled)
        ));
        assert_eq!(d.performed_installments(), 1);
        assert!(d.settle_installment(1, 7).unwrap());
        assert!(matches!(
            d.settle_installment(1, 8),
            Err(InstallmentError::AlreadySettled)
        ));
        assert!(matches!(
            d.settle_installment(9, 8),
            Err(InstallmentError::IndexOutOfRange)
        ));
    }

    /// Single renegotiation cycle.
    #[test]
    fn renegotiation_is_single_cycle() {
        let mut d = deal();
        d.advance(DealEvent::Accept, 1).unwrap();
        d.advance(DealEvent::Activate, 2).unwrap();
        d.advance(DealEvent::ConditionMet, 3).unwrap();
        d.advance(DealEvent::ObligationDue, 4).unwrap();
        assert!(d.can_renegotiate());
        d.advance(DealEvent::Renegotiate, 5).unwrap();
        assert_eq!(d.renegotiations, 1);
        assert!(!d.can_renegotiate());
        d.advance(DealEvent::Reactivate, 6).unwrap();
        assert_eq!(d.state, DealState::Active);
    }

    #[test]
    fn deal_serializes_roundtrip() {
        let mut d = deal();
        d.advance(DealEvent::Accept, 1).unwrap();
        let json = serde_json::to_string(&d).unwrap();
        let back: Deal = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
        for key in [
            "condition_deadline_minute",
            "performance_deadline_minute",
            "installments",
            "condition_overdue_emitted",
            "performance_overdue_emitted",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
    }

    /// A deal missing on old saves loads as `Proposed` via `serde(default)`
    /// on the world map; the struct itself must tolerate missing optional
    /// fields (forward migration without a version bump).
    #[test]
    fn deal_deserializes_with_defaults() {
        let minimal = serde_json::json!({
            "id": 7,
            "debtor": "a",
            "creditor": "b",
            "terms": "t",
            "condition": {"Flag": "c"},
            "performance": {"Flag": "p"},
            "state": "Proposed",
            "renegotiations": 0,
            "created_tick": 0
        });
        let d: Deal = serde_json::from_value(minimal).unwrap();
        assert_eq!(d.state, DealState::Proposed);
        assert_eq!(d.grace_minutes, 30);
        assert!(d.installments.is_empty());
    }
}
