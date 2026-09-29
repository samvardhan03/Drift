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
| `RUST_LOG` | No | `info` | Log level filter |

## Health check

```
GET /health → 200 {"status":"ok"}
```
