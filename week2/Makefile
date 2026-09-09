CARGO ?= /opt/homebrew/bin/cargo

.PHONY: reproduce
reproduce:
	$(RM) artifacts/run.json artifacts/traj.jsonl
	$(CARGO) run --release --manifest-path md/Cargo.toml -- run --out artifacts
