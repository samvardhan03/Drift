# Changelog

All notable changes are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — open-core-relaunch

### Synced from upstream drift-risk-copilot (through 765574b, 2026-10-11)
- `RiskDecomposition` now includes a GARCH(1,1) volatility forecast
- Holdings upload recognises exports from 9 Indian brokers; ISIN → NSE/BSE resolution with instrument classification
- Upstox OAuth holdings import (`/auth/upstox/login|callback|status`)
- Chart-ready visualisation data on `/ask`, led by a daily portfolio value history
- Multi-tool orchestration for `/ask`; strict trace-only narration; directional and recommendation grounding checks
- `/health` reports integration status and request load; `/ask` concurrency limiter; market-data cache warmup
- JSON errors for every rejection path

### Security
- Dependency bumps for RustSec advisories published Oct 2026: `calamine` 0.36 → 0.38 (CFB parser OOM; brings `zip` 9.5, fixing 4 extraction advisories), `rusqlite` 0.31 → 0.41 (`libsqlite3-sys` 0.39, bundled SQLite CVE), `rustls` 0.23.46, `rustls-webpki` 0.103.16, `webpki-roots` 1.0.12, `tokio` 1.54

### Changed
- Upstox redirect URI defaults to `http://localhost:8080/auth/upstox/callback` (upstream default pointed at a specific hosted deployment)
- Upstream `INTEGRATION.md` moved to `docs/api-reference.md` with placeholder hosts
- Postman collection base URL set to `http://localhost:8080`
- FAQ/README: removed an incorrect "lookahead-safe" claim for the open core and an incorrect data-client claim
- CI: `cargo fmt --check` is blocking; toolchain file declares `rustfmt`/`clippy`
- Doc comments: `experiments.rs` no longer calls `CvarRebalance` unimplemented or counts three experiments (all seven are dispatched); `cvar.rs` no longer points at a README section that doesn't exist

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
