//! Headless simulated-player runner.
//!
//! Usage:
//!   scenario-runner [--seed N] [--scenario NAME] [--verbose]
//!
//! Exits non-zero if any scenario fails, so it can gate CI.

use rpg_eval::{leaderboard, run_all, run_named};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed: u64 = 2026;
    let mut only: Option<String> = None;
    let mut verbose = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            "--scenario" => {
                i += 1;
                only = args.get(i).cloned();
            }
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                println!("scenario-runner [--seed N] [--scenario NAME] [--verbose]");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let results = match &only {
        Some(name) => match run_named(name, seed) {
            Some(r) => vec![r],
            None => {
                eprintln!("unknown scenario: {name}");
                std::process::exit(2);
            }
        },
        None => run_all(seed),
    };

    print!("{}", leaderboard(&results));
    if verbose {
        for r in &results {
            println!("\n=== {} trace ===", r.name);
            for line in &r.trace {
                println!("  {line}");
            }
        }
    }

    if results.iter().any(|r| !r.passed()) {
        std::process::exit(1);
    }
}
