//! Deterministic A* pathfinding on the navigation grid.
//!
//! Grids are small (tens of cells per side) so a simple binary-heap A* with
//! octile distance is more than fast enough, and gives byte-identical paths
//! for identical inputs.

use rpg_domain::geom::{from_grid, to_grid, GridPos, Vec2};
use rpg_domain::world::NavGrid;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct Node {
    pos: GridPos,
    /// f = g + h, stored as ordered integer-ish key for the heap.
    f: f32,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap acts as a min-heap.
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.pos.y.cmp(&self.pos.y))
            .then_with(|| other.pos.x.cmp(&self.pos.x))
    }
}

const DIRS: [(i32, i32, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

/// Returns the path from `start` to `goal` inclusive, or `None` if unreachable.
/// The start cell must be walkable; if the goal is not walkable, the closest
/// walkable neighbour is used.
pub fn astar(nav: &NavGrid, start: GridPos, goal: GridPos) -> Option<Vec<GridPos>> {
    if !nav.walkable(start) {
        return None;
    }
    let goal = if nav.walkable(goal) {
        goal
    } else {
        // Snap to nearest walkable neighbour of the (possibly wall) goal.
        DIRS.iter()
            .map(|(dx, dy, _)| GridPos::new(goal.x + dx, goal.y + dy))
            .find(|g| nav.walkable(*g))?
    };
    if start == goal {
        return Some(vec![start]);
    }

    let n = (nav.width * nav.height) as usize;
    let idx = |g: GridPos| (g.y * nav.width + g.x) as usize;
    let mut g_score = vec![f32::INFINITY; n];
    let mut came_from = vec![None::<GridPos>; n];
    let mut open = BinaryHeap::new();

    g_score[idx(start)] = 0.0;
    open.push(Node {
        pos: start,
        f: heuristic(start, goal),
    });

    while let Some(Node { pos: current, .. }) = open.pop() {
        if current == goal {
            return Some(reconstruct(&came_from, nav, current));
        }
        let cur_g = g_score[idx(current)];
        for (dx, dy, cost) in DIRS {
            let next = GridPos::new(current.x + dx, current.y + dy);
            if !nav.walkable(next) {
                continue;
            }
            // Prevent cutting diagonal corners through walls.
            if dx != 0 && dy != 0 {
                let a = GridPos::new(current.x + dx, current.y);
                let b = GridPos::new(current.x, current.y + dy);
                if !nav.walkable(a) || !nav.walkable(b) {
                    continue;
                }
            }
            let tentative = cur_g + cost;
            if tentative < g_score[idx(next)] {
                g_score[idx(next)] = tentative;
                came_from[idx(next)] = Some(current);
                open.push(Node {
                    pos: next,
                    f: tentative + heuristic(next, goal),
                });
            }
        }
    }
    None
}

fn heuristic(a: GridPos, b: GridPos) -> f32 {
    let dx = (a.x - b.x).abs() as f32;
    let dy = (a.y - b.y).abs() as f32;
    // Octile distance.
    (dx + dy) + (std::f32::consts::SQRT_2 - 2.0) * dx.min(dy)
}

fn reconstruct(came_from: &[Option<GridPos>], nav: &NavGrid, mut current: GridPos) -> Vec<GridPos> {
    let idx = |g: GridPos| (g.y * nav.width + g.x) as usize;
    let mut path = vec![current];
    while let Some(prev) = came_from[idx(current)] {
        path.push(prev);
        current = prev;
    }
    path.reverse();
    path
}

/// Plan a path to a world-space target and return the remaining waypoints in
/// world coordinates (excluding the starting cell).
pub fn plan_world_path(nav: &NavGrid, from: Vec2, to: Vec2) -> Option<Vec<Vec2>> {
    let path = astar(nav, to_grid(from), to_grid(to))?;
    Some(path.into_iter().skip(1).map(from_grid).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::world::Tile;

    #[test]
    fn straight_path_on_open_grid() {
        let nav = NavGrid::new(10, 10);
        let path = astar(&nav, GridPos::new(0, 0), GridPos::new(5, 0)).unwrap();
        assert_eq!(path.first(), Some(&GridPos::new(0, 0)));
        assert_eq!(path.last(), Some(&GridPos::new(5, 0)));
    }

    #[test]
    fn path_routes_around_wall() {
        let mut nav = NavGrid::new(7, 7);
        for y in 0..6 {
            nav.set(GridPos::new(3, y), Tile::Wall);
        }
        // Gap at y=6.
        let path = astar(&nav, GridPos::new(0, 0), GridPos::new(6, 0)).unwrap();
        assert!(path.iter().all(|g| nav.walkable(*g)));
        assert!(path.iter().any(|g| g.y >= 5), "must detour through the gap");
    }

    #[test]
    fn unreachable_returns_none() {
        let mut nav = NavGrid::new(7, 7);
        for y in 0..7 {
            nav.set(GridPos::new(3, y), Tile::Wall);
        }
        assert!(astar(&nav, GridPos::new(0, 0), GridPos::new(6, 0)).is_none());
    }

    #[test]
    fn path_is_deterministic() {
        let nav = NavGrid::new(12, 12);
        let a = astar(&nav, GridPos::new(0, 0), GridPos::new(11, 7)).unwrap();
        let b = astar(&nav, GridPos::new(0, 0), GridPos::new(11, 7)).unwrap();
        assert_eq!(a, b);
    }
}
