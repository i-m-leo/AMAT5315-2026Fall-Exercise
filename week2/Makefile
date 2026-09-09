CARGO ?= /opt/homebrew/bin/cargo

.PHONY: reproduce
reproduce:
	$(CARGO) run --release --manifest-path md/Cargo.toml -- run --out artifacts
