# Security Policy

## Supported versions

Only the latest commit on `main` receives security fixes.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email the maintainers at the address listed in the GitHub profile. Include:

- A clear description of the vulnerability
- Steps to reproduce
- Potential impact
- Your preferred disclosure timeline (default: 90 days)

We aim to acknowledge reports within 72 hours and to release a fix within 30 days for critical issues.

## Scope

This repository contains a research tool; it does not handle real money or execute trades. The primary attack surface is:

- The axum HTTP API (injection, SSRF via ticker input, path traversal in snapshot IDs)
- Dependency supply chain (cargo audit is run in CI)

API keys (`GEMINI_API_KEY` and any future keys) must never be committed to the repository. The `.gitignore` excludes `.env` files; please keep it that way.
