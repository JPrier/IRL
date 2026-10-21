# One-command developer workflow.
.PHONY: setup play text dist dist-slim verify fmt clippy test scenarios bench

# Fetch and verify every asset the game needs (neural speech + local model).
setup:
	./scripts/setup.sh

# Graphical, voice-native client with the real local model and neural voice.
# Equivalent to ./scripts/play.sh (which also runs setup on first launch).
play:
	./scripts/play.sh

# Text client (no graphics/voice), deterministic baseline.
text:
	cargo run -p game

fmt:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

scenarios:
	cargo run -q -p scenario-runner -- --seed 2026

bench:
	cargo run -p model-bench -- --speech

# Build a self-contained, ship-able bundle for this OS/arch (see README).
dist:
	./scripts/package.sh

# Same, but without the ~950 MiB of weights (fetched on first run).
dist-slim:
	./scripts/package.sh --slim

# The PR gate. Real-model/neural-voice jobs are separate (self-hosted).
verify: fmt clippy test scenarios
