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
use rpg_domain::ids::{ActorId, ItemId, LocationId, OfferId, PLAYER};
use rpg_domain::ports::InferenceEngine;
use rpg_eval::Session;
use rpg_inference::{LlamaEngine, ModelProfile};
use rpg_persistence::{load_json, restore, save_json, SaveFile};
use rpg_sim::SimConfig;
use std::io::{self, Write};

const SAVE_PATH: &str = "agentic_rpg_save.json";

fn main() {
    let opts = match Options::parse(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            eprintln!();
            usage();
            std::process::exit(2);
        }
    };
    if opts.help {
        usage();
        return;
    }
    let mut session = match build_session(&opts) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Cannot start real-model session: {e}");
            eprintln!("Pass --heuristic to run the deterministic baseline instead.");
            std::process::exit(1);
        }
    };
    announce(&session);
    println!("Seed {}. Type 'help' for commands.", opts.seed);
    println!();
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
                    report_decision(&session);
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
            "offers" | "o" => list_offers(&session),
            "accept" => {
                let id: u64 = rest.parse().unwrap_or(0);
                let out = session.player_do(Action::AcceptOffer { offer: OfferId(id) });
                println!("{out:?}");
                drain_events(&session, before);
            }
            "reject" => {
                let id: u64 = rest.parse().unwrap_or(0);
                let out = session.player_do(Action::RejectOffer { offer: OfferId(id) });
                println!("{out:?}");
                drain_events(&session, before);
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
    print_session_summary(&session);
}

// ---------------------------------------------------------------------------
// Real-model / heuristic session setup
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Options {
    seed: u64,
    profile: Option<String>,
    model: Option<String>,
    heuristic: bool,
    no_think: Option<bool>,
    n_ctx: u32,
    max_tokens: u32,
    threads: i32,
    help: bool,
}

impl Options {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut o = Options {
            seed: 2026,
            n_ctx: 4096,
            max_tokens: 192,
            ..Options::default()
        };
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--profile" => o.profile = Some(next_arg(&mut it, "--profile")?),
                "--model" => o.model = Some(next_arg(&mut it, "--model")?),
                "--heuristic" => o.heuristic = true,
                "--no-think" => o.no_think = Some(true),
                "--think" => o.no_think = Some(false),
                "--ctx" => o.n_ctx = parse_arg(&mut it, "--ctx")?,
                "--max-tokens" => o.max_tokens = parse_arg(&mut it, "--max-tokens")?,
                "--threads" => o.threads = parse_arg(&mut it, "--threads")?,
                "--seed" => o.seed = parse_arg(&mut it, "--seed")?,
                "--help" | "-h" => o.help = true,
                other if !other.starts_with('-') => {
                    o.seed = other
                        .parse()
                        .map_err(|_| format!("unexpected argument '{other}'"))?;
                }
                other => return Err(format!("unknown option '{other}'")),
            }
        }
        Ok(o)
    }
}

fn next_arg(it: &mut std::vec::IntoIter<String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_arg<T: std::str::FromStr>(
    it: &mut std::vec::IntoIter<String>,
    flag: &str,
) -> Result<T, String> {
    next_arg(it, flag)?
        .parse()
        .map_err(|_| format!("{flag} has an invalid value"))
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(16)
}

fn profile_from_options(o: &Options) -> Result<ModelProfile, String> {
    if let Some(path) = &o.profile {
        let mut p = ModelProfile::load(path).map_err(|e| format!("profile {path}: {e}"))?;
        if let Some(nt) = o.no_think {
            p.no_think = nt;
        }
        return Ok(p);
    }
    let model = o
        .model
        .as_ref()
        .ok_or_else(|| "no --profile or --model supplied".to_string())?;
    if !std::path::Path::new(model).exists() {
        return Err(format!("model file not found: {model}"));
    }
    let threads = if o.threads > 0 {
        o.threads
    } else {
        default_threads()
    };
    let path = model.replace(char::from(92), "/");
    let no_think = o.no_think.unwrap_or(true);
    let json = format!(
        r#"{{"id":"cli","model_path":"{path}","n_ctx":{ctx},"n_threads":{threads},"max_tokens":{max},"temperature":0.4,"no_think":{no_think},"deadline_ms":60000}}"#,
        path = path,
        ctx = o.n_ctx,
        threads = threads,
        max = o.max_tokens,
        no_think = no_think,
    );
    ModelProfile::from_json(&json).map_err(|e| e.to_string())
}

/// Build the session. A model is only used when one is actually requested;
/// otherwise this is explicitly the deterministic baseline.
fn build_session(opts: &Options) -> Result<Session, String> {
    if opts.heuristic || (opts.profile.is_none() && opts.model.is_none()) {
        return Ok(Session::new(opts.seed));
    }
    let profile = profile_from_options(opts)?;
    let mut engine = LlamaEngine::load(profile).map_err(|e| e.to_string())?;
    engine.warm().map_err(|e| e.to_string())?;
    Ok(Session::with_engine(opts.seed, Box::new(engine)))
}

fn announce(session: &Session) {
    println!("Agentic RPG - Hidden Mafia (text vertical slice)");
    if session.is_real_model() {
        println!(
            "Engine: {} (model {}) - real local model",
            session.engine_name(),
            session.model_id()
        );
    } else {
        println!("Engine: {} - DETERMINISTIC BASELINE", session.engine_name());
        println!("  NPC replies come from canned rules, NOT a language model.");
        println!("  Pass --profile <json> or --model <gguf> for real AI.");
    }
}

fn report_decision(session: &Session) {
    if let Some((npc, source)) = session.decisions.last() {
        if !source.is_model() {
            println!(
                "   [{npc}: {}] - deterministic answer, not model output",
                source.label()
            );
        }
    }
}

fn print_session_summary(session: &Session) {
    if session.is_real_model() {
        println!();
        println!(
            "Session: {} model decision(s), {} canonical obligation(s), {} deterministic fallback(s).",
            session.model_decisions(),
            session.canonical_decisions(),
            session.fallback_decisions()
        );
    }
}

fn usage() {
    println!(
        r#"game [options]
  --profile <file.json>  load a ModelProfile (real local model)
  --model <model.gguf>   shortcut: build a profile for a GGUF file
  --heuristic            force the deterministic baseline
  --seed <n>             scenario seed (default 2026)
  --ctx <n>              context window for --model (default 4096)
  --max-tokens <n>       max generated tokens (default 192)
  --threads <n>          CPU threads (default: available)
  --no-think / --think   override the reasoning block
  --help"#
    );
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
         \x20 investigate <id>            search a location or person (pick up items)\n\
         \x20 give <npc_id> <item_id>     hand over an item\n\
         \x20 threaten <npc_id> <demand>  make a threat (has consequences)\n\
         \x20 offers                      list pending offers\n\
         \x20 accept <offer_id>           accept a pending offer\n\
         \x20 reject <offer_id>           reject a pending offer\n\
         \x20 status                      money, beliefs, relationships, deals\n\
         \x20 wait <n>                    advance n simulation ticks\n\
         \x20 save [path] / load [path]   persist or restore the world\n\
         \x20 quit"
    );
}

fn list_offers(s: &Session) {
    let player = ActorId::new(PLAYER);
    let offers: Vec<_> = s
        .sim
        .world
        .offers
        .values()
        .filter(|o| {
            o.state == rpg_domain::social::OfferState::Pending
                && (o.to == player || o.from == player)
        })
        .collect();
    if offers.is_empty() {
        println!("No pending offers.");
        return;
    }
    for o in offers {
        let want: Vec<String> = o
            .want
            .iter()
            .map(|s| format!("{}x{}", s.qty, s.item))
            .collect();
        let give: Vec<String> = o
            .give
            .iter()
            .map(|s| format!("{}x{}", s.qty, s.item))
            .collect();
        println!(
            "#{} from {} to {}: give [{}], want [{}], condition {:?} — {}",
            o.id.0,
            o.from,
            o.to,
            give.join(", "),
            want.join(", "),
            o.condition,
            o.note
        );
    }
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
