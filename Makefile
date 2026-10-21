# One-command developer workflow.
.PHONY: verify fmt clippy test scenarios bench play

fmt:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

scenarios:
	cargo run -q -p scenario-runner -- --seed 2026

bench:
	cargo run -q -p model-bench -- --speech

play:
	cargo run -p game

# The PR gate. Real-model/neural-voice jobs are separate (self-hosted).
verify: fmt clippy test scenarios
