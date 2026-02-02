# Pivota Glow Agent (Orchestrator/BFF)

FastAPI orchestrator for the `pivota-glow-guide` frontend.

This service exposes a stable API surface for the frontend:

- `POST /v1/diagnosis`
- `POST /v1/photos` (multipart)
- `GET /v1/photos/qc?upload_id=...` (poll QC for pending uploads)
- `POST /v1/photos/sample`
- `POST /v1/events` (client analytics ingest; optional PostHog forward)
- `POST /v1/analysis`
- `POST /v1/analysis/risk`
- `POST /v1/routine/reorder`
- `PATCH /v1/routine/selection`
- `POST /v1/checkout`
- `POST /v1/affiliate/outcome`

Internally it can call:

- Pivota Agent Gateway (product search / quotes / orders) – Railway
- Aurora Decision service (optional) – Vercel

## Local dev

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export CORS_ORIGINS="http://localhost:5173,https://pivota-glow-guide.vercel.app"
export PIVOTA_AGENT_GATEWAY_BASE_URL="https://pivota-agent-production.up.railway.app"
export AURORA_DECISION_BASE_URL="https://aurora-beauty-decision-system.vercel.app"

uvicorn app.main:app --reload --port 8080
```

## Deploy (Railway)

### 1) Create project

- Railway → **New Project** → **Deploy from GitHub repo**
- Select `pengxu9-rgb/Pivota-glow-agent`

### 2) Set variables (Railway → Variables)

- `CORS_ORIGINS` = `https://pivota-glow-guide.vercel.app`
- `PIVOTA_AGENT_GATEWAY_BASE_URL` = `https://pivota-agent-production.up.railway.app`
- `PIVOTA_AGENT_API_KEY` = *(optional; if your gateway requires it)*
- `AURORA_DECISION_BASE_URL` = `https://aurora-beauty-decision-system.vercel.app`
- `LOG_LEVEL` = `INFO`
- `POSTHOG_API_KEY` / `POSTHOG_HOST` = *(optional; forwards `/v1/events` to PostHog)*

### 3) Start command

Railway should pick up `railway.json`. If you set it manually, use:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 4) Wire the frontend

In `pivota-glow-guide` (Vercel → Environment Variables):

- `VITE_API_BASE_URL` = `https://<your-glow-agent-railway-domain>/v1`
- `VITE_UPLOAD_ENDPOINT` = *(leave unset; it will fall back to `VITE_API_BASE_URL`)*

Then redeploy `pivota-glow-guide`.
