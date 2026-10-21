//! # rpg_perception
//!
//! Deterministic spatial perception: line of sight, field of view, sound
//! propagation and speech routing (addressee + overhearers).
//!
//! The key contract is that an observer receives a **subjective perception
//! event**, never the global truth. This crate computes *who could perceive
//! what*, and nothing else.

use rpg_domain::action::Loudness;
use rpg_domain::geom::{to_grid, GridPos, Vec2};
use rpg_domain::ids::{ActorId, LocationId};
use rpg_domain::world::{NavGrid, World};

/// Tunable perception parameters. Fixed per scenario so replays are stable.
#[derive(Clone, Debug)]
pub struct PerceptionConfig {
    pub fov_degrees: f32,
    pub sight_range: f32,
    /// Fraction of remaining audible radius lost per wall crossed.
    pub hearing_wall_penalty: f32,
    /// Minimum effective audible radius (never fully mute through one wall).
    pub min_audible_radius: f32,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            fov_degrees: 120.0,
            sight_range: 14.0,
            hearing_wall_penalty: 0.55,
            min_audible_radius: 0.75,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerceptionEngine {
    pub config: PerceptionConfig,
}

/// Result of routing a spoken utterance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechRouting {
    pub addressee: Option<ActorId>,
    pub overhearers: Vec<ActorId>,
    /// True when the addressee was inferred rather than explicitly supplied.
    pub inferred: bool,
}

impl Default for PerceptionEngine {
    fn default() -> Self {
        Self::new(PerceptionConfig::default())
    }
}

impl PerceptionEngine {
    pub fn new(config: PerceptionConfig) -> Self {
        Self { config }
    }

    /// Bresenham line of grid cells from `a` to `b`, inclusive.
    pub fn line_cells(a: GridPos, b: GridPos) -> Vec<GridPos> {
        let mut cells = Vec::new();
        let (mut x0, mut y0) = (a.x, a.y);
        let (x1, y1) = (b.x, b.y);
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            cells.push(GridPos::new(x0, y0));
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
        cells
    }

    /// Count cells that block sight strictly between the endpoints.
    pub fn sight_blockers(&self, nav: &NavGrid, from: Vec2, to: Vec2) -> usize {
        let cells = Self::line_cells(to_grid(from), to_grid(to));
        if cells.len() <= 2 {
            return 0;
        }
        cells[1..cells.len() - 1]
            .iter()
            .filter(|c| nav.tile(**c).map(|t| t.blocks_sight()).unwrap_or(true))
            .count()
    }

    pub fn has_line_of_sight(&self, nav: &NavGrid, from: Vec2, to: Vec2) -> bool {
        self.sight_blockers(nav, from, to) == 0
    }

    pub fn in_field_of_view(&self, from: Vec2, facing: f32, to: Vec2) -> bool {
        let dir = from.direction_to(to);
        if dir == Vec2::ZERO {
            return true;
        }
        let facing_dir = Vec2::new(facing.cos(), facing.sin());
        let dot = facing_dir.dot(dir).clamp(-1.0, 1.0);
        let angle = dot.acos().to_degrees();
        angle <= self.config.fov_degrees * 0.5
    }

    /// Can `observer` (position + facing) visually see `target_pos`?
    pub fn can_see(&self, nav: &NavGrid, from: Vec2, facing: f32, to: Vec2) -> bool {
        if from.distance(to) > self.config.sight_range {
            return false;
        }
        self.in_field_of_view(from, facing, to) && self.has_line_of_sight(nav, from, to)
    }

    /// Effective audible radius of a sound after wall attenuation.
    pub fn effective_audible_radius(
        &self,
        nav: &NavGrid,
        from: Vec2,
        to: Vec2,
        loudness: Loudness,
        source_transmission: f32,
    ) -> f32 {
        let base = loudness.radius() * source_transmission.max(0.05);
        let walls = self.wall_count(nav, from, to);
        let mut r = base;
        for _ in 0..walls {
            r *= 1.0 - self.config.hearing_wall_penalty;
        }
        r.max(self.config.min_audible_radius)
    }

    /// Number of cells that block sound between two points (endpoints inclusive
    /// of solid cells is avoided; interior only).
    pub fn wall_count(&self, nav: &NavGrid, from: Vec2, to: Vec2) -> usize {
        let cells = Self::line_cells(to_grid(from), to_grid(to));
        if cells.len() <= 2 {
            return 0;
        }
        cells[1..cells.len() - 1]
            .iter()
            .filter(|c| nav.tile(**c).map(|t| t.blocks_sound()).unwrap_or(true))
            .count()
    }

    pub fn can_hear(
        &self,
        nav: &NavGrid,
        from: Vec2,
        to: Vec2,
        loudness: Loudness,
        source_transmission: f32,
    ) -> bool {
        let dist = from.distance(to);
        let radius = self.effective_audible_radius(nav, from, to, loudness, source_transmission);
        dist <= radius
    }

    /// Compute the addressee and overhearers for an utterance.
    ///
    /// * `explicit` — the addressee the player/model named, if any.
    /// * Falls back to the nearest living actor within the speaker's FOV and
    ///   normal speaking range when no explicit target is supplied.
    #[allow(clippy::too_many_arguments)]
    pub fn route_speech(
        &self,
        world: &World,
        speaker: &ActorId,
        explicit: Option<&ActorId>,
        loudness: Loudness,
        source_pos: Vec2,
        source_facing: f32,
        source_location: &LocationId,
    ) -> SpeechRouting {
        let transmission = world
            .location(source_location)
            .map(|l| l.sound_transmission)
            .unwrap_or(1.0);
        let nav = &world.nav;

        let mut addressee = explicit.cloned();
        let mut inferred = false;

        let audible = |a: &rpg_domain::world::Actor| -> bool {
            a.is_alive()
                && a.id != *speaker
                && self.can_hear(nav, source_pos, a.pos, loudness, transmission)
        };

        if addressee.is_none() {
            // Infer from facing + proximity among audible actors.
            let mut best: Option<(f32, ActorId)> = None;
            for a in world.actors.values() {
                if !audible(a) {
                    continue;
                }
                if !self.in_field_of_view(source_pos, source_facing, a.pos) {
                    continue;
                }
                let d = source_pos.distance(a.pos);
                if best.as_ref().map(|(bd, _)| d < *bd).unwrap_or(true) {
                    best = Some((d, a.id.clone()));
                }
            }
            if let Some((_, id)) = best {
                addressee = Some(id);
                inferred = true;
            }
        }

        // If an explicit addressee is out of range, treat as unaddressed.
        if let Some(t) = addressee.clone() {
            let ok = world
                .actor(&t)
                .map(|a| {
                    a.is_alive() && self.can_hear(nav, source_pos, a.pos, loudness, transmission)
                })
                .unwrap_or(false);
            if !ok {
                addressee = None;
                inferred = false;
            }
        }

        let mut overhearers: Vec<ActorId> = world
            .actors
            .values()
            .filter(|a| audible(a))
            .map(|a| a.id.clone())
            .filter(|id| Some(id) != addressee.as_ref())
            .collect();
        overhearers.sort();

        SpeechRouting {
            addressee,
            overhearers,
            inferred,
        }
    }

    /// All actors who can see a point (used for witnessing physical events).
    pub fn seers(&self, world: &World, pos: Vec2, exclude: &ActorId) -> Vec<ActorId> {
        let mut out: Vec<ActorId> = world
            .actors
            .values()
            .filter(|a| a.id != *exclude && a.is_alive())
            .filter(|a| self.can_see(&world.nav, a.pos, a.facing, pos))
            .map(|a| a.id.clone())
            .collect();
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::geom::Vec2;

    fn open_grid() -> NavGrid {
        NavGrid::new(20, 20)
    }

    #[test]
    fn los_is_clear_on_open_floor() {
        let p = PerceptionEngine::default();
        let nav = open_grid();
        assert!(p.has_line_of_sight(&nav, Vec2::new(1.5, 1.5), Vec2::new(5.5, 1.5)));
    }

    #[test]
    fn wall_blocks_line_of_sight() {
        let p = PerceptionEngine::default();
        let mut nav = open_grid();
        nav.set(GridPos::new(3, 1), rpg_domain::world::Tile::Wall);
        assert!(!p.has_line_of_sight(&nav, Vec2::new(1.5, 1.5), Vec2::new(5.5, 1.5)));
    }

    #[test]
    fn facing_cone_limits_vision() {
        let p = PerceptionEngine::default();
        let nav = open_grid();
        // Facing +x, target behind at -x should be outside a 120° cone.
        assert!(!p.can_see(&nav, Vec2::new(5.0, 5.0), 0.0, Vec2::new(1.0, 5.0)));
        assert!(p.can_see(&nav, Vec2::new(5.0, 5.0), 0.0, Vec2::new(9.0, 5.0)));
    }

    #[test]
    fn whisper_is_quieter_than_shout() {
        let p = PerceptionEngine::default();
        let nav = open_grid();
        let from = Vec2::new(1.0, 1.0);
        let to = Vec2::new(5.0, 1.0);
        assert!(p.can_hear(&nav, from, to, Loudness::Shout, 1.0));
        assert!(!p.can_hear(&nav, from, to, Loudness::Whisper, 1.0));
    }

    #[test]
    fn line_cells_are_contiguous() {
        let cells = PerceptionEngine::line_cells(GridPos::new(0, 0), GridPos::new(5, 3));
        assert_eq!(cells.first(), Some(&GridPos::new(0, 0)));
        assert_eq!(cells.last(), Some(&GridPos::new(5, 3)));
        for w in cells.windows(2) {
            assert!((w[0].x - w[1].x).abs() <= 1 && (w[0].y - w[1].y).abs() <= 1);
        }
    }
}
