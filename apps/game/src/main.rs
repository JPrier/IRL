//! Text-mode playable client for the Hidden Mafia vertical slice.
//!
//! This is a *client* over the same canonical simulation the headless tests
//! use. It deliberately uses the deterministic `HeuristicEngine` fallback so
//! it runs with zero model assets; plugging in a real local LLM only requires
//! constructing a different `InferenceEngine` in `Session`.
//!
//! Commands: look, go, talk, private, investigate, give, threaten, status,
//!           wait, save, load, help, quit

use rpg_domain::action::{Action, InvestigateTarget, ValidationOutcome};
use rpg_domain::ids::{ActorId, ItemId, LocationId, PLAYER};
use rpg_eval::Session;
use rpg_persistence::{load_json, restore, save_json, SaveFile};
use rpg_sim::SimConfig;
use std::io::{self, Write};

const SAVE_PATH: &str = "agentic_rpg_save.json";

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026);
    let mut session = Session::new(seed);
    println!("Agentic RPG — Hidden Mafia (text vertical slice)");
    println!("Seed {seed}. Type 'help' for commands.\n");
    look(&session);

    loop {
        print!("\n> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();

        let before = session.sim.events.len();
        match cmd {
            "look" | "l" => look(&session),
            "help" | "h" | "?" => help(),
            "quit" | "q" | "exit" => {
                println!("Goodbye.");
                break;
            }
            "go" => {
                if rest.is_empty() {
                    println!("Usage: go <location_id>");
                } else {
                    let out = session.player_do(Action::MoveTo {
                        location: LocationId::new(rest),
                    });
                    println!("{out:?}");
                    if matches!(out, ValidationOutcome::Accepted) {
                        // Advance until the player arrives (bounded).
                        let player = ActorId::new(PLAYER);
                        let mut guard = 0;
                        while session.sim.paths.contains_key(&player) && guard < 500 {
                            let _ = session.tick();
                            guard += 1;
                        }
                    }
                    look(&session);
                }
            }
            "talk" | "say" => {
                let (npc, text) = split_npc_text(rest);
                if npc.is_empty() || text.is_empty() {
                    println!("Usage: talk <npc_id> <what to say>");
                } else {
                    session.say(npc, text);
                    drain_events(&session, before);
                }
            }
            "private" => {
                if rest.is_empty() {
                    println!("Usage: private <npc_id>");
                } else {
                    let out = session.player_do(Action::AskPrivately {
                        target: ActorId::new(rest),
                    });
                    println!("{out:?}");
                    look(&session);
                }
            }
            "investigate" | "inv" => {
                if rest.is_empty() {
                    println!("Usage: investigate <location_id|npc_id>");
                } else {
                    let target = if session
                        .sim
                        .world
                        .locations
                        .contains_key(&LocationId::new(rest))
                    {
                        InvestigateTarget::Location(LocationId::new(rest))
                    } else if session.sim.world.actors.contains_key(&ActorId::new(rest)) {
                        InvestigateTarget::Actor(ActorId::new(rest))
                    } else {
                        InvestigateTarget::Item(ItemId::new(rest))
                    };
                    let out = session.player_do(Action::Investigate { target });
                    println!("{out:?}");
                    drain_events(&session, before);
                }
            }
            "give" => {
                let (npc, item) = split_npc_text(rest);
                if npc.is_empty() || item.is_empty() {
                    println!("Usage: give <npc_id> <item_id>");
                } else {
                    let out = session.player_do(Action::TransferItem {
                        target: ActorId::new(npc),
                        item: ItemId::new(item),
                        qty: 1,
                    });
                    println!("{out:?}");
                    drain_events(&session, before);
                }
            }
            "threaten" => {
                let (npc, demand) = split_npc_text(rest);
                if npc.is_empty() || demand.is_empty() {
                    println!("Usage: threaten <npc_id> <demand>");
                } else {
                    let out = session.player_do(Action::Threaten {
                        target: ActorId::new(npc),
                        demand: demand.into(),
                    });
                    println!("{out:?}");
                    drain_events(&session, before);
                }
            }
            "status" | "s" => status(&session),
            "wait" | "w" => {
                let n: u32 = rest.parse().unwrap_or(1);
                session.ticks(n);
                println!(
                    "Advanced {n} tick(s). It is now {}.",
                    session.sim.world.clock_string()
                );
                drain_events(&session, before);
            }
            "save" => {
                let path = if rest.is_empty() { SAVE_PATH } else { rest };
                let save = SaveFile::from_sim(&session.sim);
                match save_json(&save, path) {
                    Ok(()) => println!("Saved to {path}."),
                    Err(e) => println!("Save failed: {e}"),
                }
            }
            "load" => {
                let path = if rest.is_empty() { SAVE_PATH } else { rest };
                match load_json(path) {
                    Ok(save) => {
                        session.sim = restore(save, SimConfig::default());
                        println!("Loaded {path}.");
                        look(&session);
                    }
                    Err(e) => println!("Load failed: {e}"),
                }
            }
            other => println!("Unknown command '{other}'. Type 'help'."),
        }
    }
}

fn split_npc_text(rest: &str) -> (&str, &str) {
    let mut it = rest.splitn(2, ' ');
    let npc = it.next().unwrap_or("");
    let text = it.next().unwrap_or("").trim();
    (npc, text)
}

fn help() {
    println!(
        "Commands:\n\
         \x20 look                       describe the current room and people\n\
         \x20 go <location_id>            walk to a location\n\
         \x20 talk <npc_id> <text>        speak; the NPC decides and responds\n\
         \x20 private <npc_id>            ask to talk somewhere quiet\n\
         \x20 investigate <id>            search a location or person\n\
         \x20 give <npc_id> <item_id>     hand over an item\n\
         \x20 threaten <npc_id> <demand>  make a threat (has consequences)\n\
         \x20 status                      money, beliefs, relationships, deals\n\
         \x20 wait <n>                    advance n simulation ticks\n\
         \x20 save [path] / load [path]   persist or restore the world\n\
         \x20 quit"
    );
}

fn look(s: &Session) {
    let w = &s.sim.world;
    let p = s.player();
    let loc_name = w
        .location(&p.location)
        .map(|l| l.name.clone())
        .unwrap_or_else(|| p.location.to_string());
    println!("\n== {loc_name} ({}) ==", w.clock_string());
    let here: Vec<String> = w
        .actors
        .values()
        .filter(|a| a.is_alive() && a.location == p.location && a.id.as_str() != PLAYER)
        .map(|a| format!("{} [{}]", a.name, a.id))
        .collect();
    if here.is_empty() {
        println!("No one else is here.");
    } else {
        println!("Here: {}", here.join(", "));
    }
    let mut exits: Vec<String> = w
        .locations
        .values()
        .filter(|l| l.id != p.location)
        .map(|l| l.id.to_string())
        .collect();
    exits.sort();
    println!("Locations: {}", exits.join(", "));
}

fn drain_events(s: &Session, from: usize) {
    for e in &s.sim.events.all()[from..] {
        let desc = e.describe();
        // Keep the console readable: hide bare ticks.
        if !desc.starts_with("tick") {
            println!("   - {desc}");
        }
    }
}

fn status(s: &Session) {
    let w = &s.sim.world;
    let p = s.player();
    println!("\n--- Status ({}) ---", w.clock_string());
    println!("Money: {}", p.inventory.money);
    if p.inventory.items.is_empty() {
        println!("Inventory: (empty)");
    } else {
        let items: Vec<String> = p
            .inventory
            .items
            .iter()
            .map(|s| format!("{}x{}", s.qty, s.item))
            .collect();
        println!("Inventory: {}", items.join(", "));
    }
    let beliefs: Vec<String> = p
        .beliefs
        .iter()
        .filter(|(_, b)| b.believes_true)
        .map(|(f, _)| f.to_string())
        .collect();
    println!(
        "Known facts: {}",
        if beliefs.is_empty() {
            "(none)".to_string()
        } else {
            beliefs.join(", ")
        }
    );
    println!("Deals / obligations:");
    if w.commitments.is_empty() {
        println!("  (none)");
    } else {
        for c in w.commitments.values() {
            println!(
                "  #{} [{:?}] {} owes {}: {}",
                c.id, c.state, c.debtor, c.creditor, c.terms
            );
        }
    }
}
