# Contributing to Drift

## Quick start

```bash
git clone https://github.com/samvardhan03/Drift.git
cd Drift
cargo build --workspace
cargo test --workspace
```

## Ground rules

- Every pull request must pass `cargo fmt --check`, `cargo clippy -D warnings`, and `cargo test --workspace`.
- New compute logic must have a unit test using synthetic data (no network calls in tests).
- Do not add formulas, constants, or algorithms sourced from the proprietary Drift Engine. Open-source contributions must be independently derived.
- Keep the evidence-trace contract stable; breaking changes to `EvidenceTrace` fields require a deprecation notice in CHANGELOG.md.

## Opening issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Bug reports need a minimal reproducer. Feature requests need a one-sentence "why" before the "what".

## Licensing

By contributing, you agree your changes are licensed under the same MIT OR Apache-2.0 dual license as the project. The `NOTICE` file explains what is open and what is proprietary.

## Code of conduct

See `CODE_OF_CONDUCT.md`.
