# Coze Pareto Optimizer (Minimal)

This repository contains only the multi-objective optimization plugin service:

- `GET /api/plugin/pareto/health`
- `POST /api/plugin/pareto/optimize`

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deploy on Render

This repo includes `render.yaml`, so you can deploy as a Web Service directly from GitHub.

## Coze import

After Render gives you a fixed HTTPS URL:

1. Update `coze_plugin/pareto_optimizer_openapi.yaml` `servers.url`.
2. Import that OpenAPI file in Coze plugin.

