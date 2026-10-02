# vercel-deploys

Public HTTP endpoints for TDS (Tools in Data Science, IITM BS) assignments, served by
one FastAPI app on Vercel.

The app sits at the **repo root** and deploys from **`main`**, so Vercel needs no Root
Directory or Production Branch settings: import, deploy, done. Notes on how each
question is graded live in [`sreekarsarma55/notes`](https://github.com/sreekarsarma55/notes/tree/tds)
on the `tds` branch.

## Endpoints

| Path | Method | For |
|---|---|---|
| `/` | GET | health check: `{"status": "ok", ...}` |
| `/code-interpreter` | POST | GA0 Q5: Code Interpreter with AI Error Analysis |

```text
POST /code-interpreter
{"code": "x = 10\ny = 0\nresult = x / y"}

200 application/json
{"error": [3], "result": "Traceback (most recent call last):\n  File \"<code>\", line 3, in <module>\nZeroDivisionError: division by zero\n"}
```

CORS is open (`Access-Control-Allow-Origin: *`) because the graders call these endpoints
from a browser page.

## Layout

```text
api/index.py       FastAPI app (Vercel's Python runtime serves the `app` object)
requirements.txt   runtime dependencies
vercel.json        routes every path to api/index.py
```

## Deploy

1. <https://vercel.com/new>, then **Import** `sreekarsarma55/vercel-deploys`.
2. Leave every setting at its default and click **Deploy**.
3. Every later push to `main` redeploys automatically.

## Environment variables (optional)

| Name | Used by | Effect |
|---|---|---|
| `AIPIPE_TOKEN` | `/code-interpreter` | enables the AI error analyst; responses gain `ai_error_lines` and `ai_agreed` |
| `AIPIPE_MODEL` | `/code-interpreter` | model for the analyst, default `gpt-4.1-nano` |

Without a token the endpoint still reports error lines, taken from the real traceback.

## Run locally

```bash
uv run --with fastapi --with uvicorn python api/index.py
# http://127.0.0.1:8000/code-interpreter
```
