# Changelog

All notable changes are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — open-core-relaunch

### Added
- Complete Rust workspace: `crates/agent`, `crates/compute`, `crates/server`, `crates/store`
- Axum HTTP API: `/experiment`, `/ask`, `/report/:id`, `/execution-trace/:id`, `/scenarios`, `/health`, `/portfolio/upload`
- Embedded single-page UI
- SQLite snapshot store for risk baselines and PDF reports
- PDF report generation from `EvidenceTrace`
- Graceful degraded startup when `GEMINI_API_KEY` is absent (compute-only mode)
- CI workflow: fmt check, clippy, tests, cargo-audit
- `NOTICE` file clarifying open vs. proprietary boundary
- `docs/architecture.md`, `docs/deploy.md`, `docs/faq.md`
- Issue/PR templates, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- Dual MIT OR Apache-2.0 license for all new code; copilot MIT notice preserved

### Removed
- Python showcase (FastAPI + Streamlit fixture shell) — preserved at tag `v0-python-showcase`
- GCP-specific `cloudrun.yaml` and `deploy.yml` — replaced by `docs/deploy.md`

## [v0-python-showcase]

Initial public Python fixture showcase (FastAPI + Streamlit, precomputed fixtures, no real analytics).
