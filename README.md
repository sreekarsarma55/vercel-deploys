# vercel-deploys

Public HTTP endpoints for TDS (Tools in Data Science, IITM BS) assignments, served by
one FastAPI app on Vercel.

The app sits at the **repo root** and deploys from **`main`**, so Vercel needs no Root
Directory or Production Branch settings. Every push to `main` redeploys automatically.
Notes on how each question is graded live in
[`sreekarsarma55/notes`](https://github.com/sreekarsarma55/notes/tree/tds) on the `tds` branch.

Live: <https://vercel-deploys-iota.vercel.app/>

## Endpoints

| Path | Method | GA0 question |
|---|---|---|
| `/` | GET | health check, lists the endpoints |
| `/code-interpreter` | POST | Q5: Code Interpreter with AI Error Analysis |
| `/api` | GET | Q10: FastAPI server to serve data |
| `/api/latency` | POST | Q25: POST analytics endpoint on Vercel |

CORS is open (`Access-Control-Allow-Origin: *`) because the graders call these
endpoints from a browser page.

### `POST /code-interpreter`

```text
{"code": "x = 10\ny = 0\nresult = x / y"}
-> {"error": [3], "result": "Traceback (most recent call last):\n  File \"<code>\", line 3, in <module>\nZeroDivisionError: division by zero\n"}
```

Runs the code, returns exact stdout or the traceback, and reports the line that raised.

### `GET /api?class=1A&class=1B`

```text
-> {"students": [{"studentId": 4, "class": "1A"}, ...]}
```

Students from `api/data/q-fastapi.csv`. With no `class` parameter it returns all 2000.
`class` may repeat; results stay in CSV order, not the order the classes were asked for.

### `POST /api/latency`

```text
{"regions": ["apac", "emea"], "threshold_ms": 164}
-> {"regions": [{"region": "apac", "avg_latency": 156.69, "p95_latency": 213.65,
                 "avg_uptime": 98.406, "breaches": 5}, ...]}
```

Per-region stats from `api/data/q-vercel-latency.json`. `p95_latency` uses linear
interpolation (the same as `numpy.percentile`'s default). `breaches` counts records with
`latency_ms` strictly above the threshold.

## Layout

```text
api/index.py                    FastAPI app (Vercel serves the `app` object)
api/data/q-fastapi.csv          Q10 data
api/data/q-vercel-latency.json  Q25 data
requirements.txt                runtime dependencies
vercel.json                     routes every path to api/index.py, bundles api/data/
email.json                      GA0 Q24 (Use GitHub)
.github/workflows/smoke-test.yml  GA0 Q13 (GitHub Action), also tests every endpoint
```

The data files are generated per student by the quiz, so they're specific to this
account's GA0.

## Deploy

1. <https://vercel.com/new>, then **Import** `sreekarsarma55/vercel-deploys`.
2. Leave every setting at its default and click **Deploy**.

## Environment variables (optional)

| Name | Used by | Effect |
|---|---|---|
| `AIPIPE_TOKEN` | `/code-interpreter` | enables the AI error analyst; responses gain `ai_error_lines` and `ai_agreed` |
| `AIPIPE_MODEL` | `/code-interpreter` | model for the analyst, default `gpt-4.1-nano` |

Without a token the endpoint still reports error lines, taken from the real traceback.

## Run locally

```bash
uv run --with fastapi --with uvicorn python api/index.py
# http://127.0.0.1:8000/
```
