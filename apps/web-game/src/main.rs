#![recursion_limit = "256"]
//! IRL in the browser.
//!
//! The same game as `bevy-game`: the shared `irl_client` presentation drawing
//! snapshots of the shared authoritative `SimCore`, which here is driven from
//! the browser event loop by `rpg_web_runtime` instead of native threads.
//! NPC decisions come from OpenRouter with the player's own key (or the
//! deterministic policy without one); every proposal goes through the same
//! contract validation, canonical policy and simulation commit as native.
//!
//! Build with `./scripts/web-build.sh`; see `docs/web.md`.

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
fn main() {
    web::run();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "web-game is IRL's browser build and only runs as WebAssembly.\n\
         Build and serve it with ./scripts/web-dev.sh (or scripts/web-dev.ps1),\n\
         or run the native game with `cargo run -p bevy-game`."
    );
    std::process::exit(2);
}
