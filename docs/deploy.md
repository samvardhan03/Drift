# Deployment Guide

## Prerequisites

- Docker
- A `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/) (free tier is sufficient for development)
- Optional: `SNAPSHOT_DB_PATH` — path for the SQLite snapshot store (default: `./data/snapshots.db`)

## Local (Docker)

```bash
docker build -t drift .
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key_here \
  -e SNAPSHOT_DB_PATH=/data/snapshots.db \
  -v $(pwd)/data:/data \
  drift
```

Open http://localhost:8080 in your browser.

**Without a Gemini key** — the server still starts in compute-only mode:

```bash
docker run -p 8080:8080 drift
# /experiment returns EvidenceTrace; /ask returns 503
```

## Local (cargo)

```bash
export GEMINI_API_KEY=your_key_here   # optional
export SNAPSHOT_DB_PATH=./data/snapshots.db
cargo run -p server --release
```

## Cloud Run (example)

Replace `PROJECT_ID` and `REGION` with your values.

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/drift:latest

# Deploy
gcloud run deploy drift \
  --image gcr.io/PROJECT_ID/drift:latest \
  --platform managed \
  --region REGION \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars SNAPSHOT_DB_PATH=/data/snapshots.db \
  --set-secrets GEMINI_API_KEY=drift-gemini-key:latest
```

> **Persistence note:** Cloud Run's local disk is ephemeral (lost on cold start / scale-to-zero). For persistent snapshots in production, point `SNAPSHOT_DB_PATH` at a Cloud SQL instance or a mounted volume.

## Fly.io (example)

```bash
fly launch --name drift --dockerfile Dockerfile
fly secrets set GEMINI_API_KEY=your_key_here
fly deploy
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | No (AI features only) | — | Google Gemini API key |
| `SNAPSHOT_DB_PATH` | No | `./data/snapshots.db` | SQLite path for risk snapshots |
| `PORT` | No | `8080` | Listen port |
| `UPSTOX_API_KEY` / `UPSTOX_API_SECRET` | No | — | Enables Upstox OAuth holdings import |
| `UPSTOX_REDIRECT_URI` | No | `http://localhost:8080/auth/upstox/callback` | Set to `https://YOUR_HOST/auth/upstox/callback` in deployment; must match the URL registered with Upstox |
| `RUST_LOG` | No | `info` | Log level filter |

Store `GEMINI_API_KEY` and `UPSTOX_API_SECRET` in your platform's secret manager, never in the image or repository.

## Scaling note

Snapshots live in a per-instance SQLite file. If you run more than one instance, a trace stored by one instance isn't visible to another (`/report/:id` and `/drift` may 404). Either run a single instance or move the store to shared storage.

## Health check

```
GET /health → 200
{"status":"ok","version":"0.1.0","gemini":"configured|not_configured",
 "upstox":"configured|not_configured","active_requests":0,"capacity":N}
```
