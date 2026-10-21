//! Stable, serializable identifiers for every addressable entity in the world.
//!
//! IDs are deliberately *content-stable strings* (or monotonically increasing
//! integers for log-only records) so that save files, event logs and replay
//! remain valid across code changes.  Never key gameplay on Rust collection
//! order or Bevy entity ids.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! string_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

macro_rules! num_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl $name {
            pub const ZERO: Self = Self(0);
            pub fn next(self) -> Self {
                Self(self.0 + 1)
            }
            pub fn raw(self) -> u64 {
                self.0
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

string_id!(
    ActorId,
    "Identifies an actor (player, named NPC or background actor)."
);
string_id!(
    LocationId,
    "Identifies a named location / room in the town."
);
string_id!(ItemId, "Identifies an item prototype/type.");
string_id!(
    FactionId,
    "Identifies a faction, business or organised group."
);
string_id!(
    FactId,
    "Identifies a canonical or candidate fact about the world."
);

num_id!(EventId, "Monotonic id of a committed canonical event.");
num_id!(OfferId, "Id of a proposed or accepted exchange.");
num_id!(CommitmentId, "Id of a promise, favour, debt or obligation.");
num_id!(RumorId, "Id of a rumor or claim in circulation.");

/// Convenience: the well-known player id.
pub const PLAYER: &str = "player";
