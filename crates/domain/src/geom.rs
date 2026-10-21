//! Small 2D geometry helpers. Values are in *world units* (≈ metres) unless a
//! type explicitly says otherwise. All maths is deterministic float64-free
//! where it matters for replay: positions are `f32`, but every simulation
//! decision uses these helpers directly so replays are reproducible.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Vec2) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn distance_sq(self, other: Vec2) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Unit vector from `self` toward `other`; zero when coincident.
    pub fn direction_to(self, other: Vec2) -> Vec2 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len <= f32::EPSILON {
            Vec2::ZERO
        } else {
            Vec2::new(dx / len, dy / len)
        }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(self) -> Vec2 {
        let len = self.length();
        if len <= f32::EPSILON {
            Vec2::ZERO
        } else {
            Vec2::new(self.x / len, self.y / len)
        }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn scaled(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }
}

/// A position on the integer navigation grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridPos {
    pub x: i32,
    pub y: i32,
}

impl GridPos {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Chebyshev-ish octile distance used for navigation heuristics.
    pub fn octile(self, other: GridPos) -> i32 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        dx.max(dy)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x <= self.x + self.w && p.y >= self.y && p.y <= self.y + self.h
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

/// Convert a continuous position into a grid cell (1 unit per cell).
pub fn to_grid(p: Vec2) -> GridPos {
    GridPos::new(p.x.floor() as i32, p.y.floor() as i32)
}

/// Convert a grid cell into the centre of that cell in world units.
pub fn from_grid(g: GridPos) -> Vec2 {
    Vec2::new(g.x as f32 + 0.5, g.y as f32 + 0.5)
}
