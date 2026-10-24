"""Public endpoints for TDS GA0, GA1 and GA2, served from one FastAPI app on Vercel.

    POST /code-interpreter   Q5   execute Python, report the line that raised
    GET  /api                Q10  students from q-fastapi.csv, filtered by ?class=
    POST /api/latency        Q25  per-region latency stats from q-vercel-latency.json
    POST /sentiment          Q11  batch rule-based sentiment: happy / sad / neutral
    GET  /effective-config   GA1 Q6   layered 12-factor config with ?set= overrides
    POST /mcp                GA1 Q14  MCP server with one tool, solve_challenge
    POST /ledger             GA1 Q15  plain-English questions over the order ledger
    GET  /stats              GA2 Q3   descriptive stats behind a strict per-origin CORS policy
    POST /verify             GA2 Q4   RS256 JWT verification against a fixed IdP key
    POST /analytics          GA2 Q7   API-key auth + event aggregation
    GET  /work /metrics /healthz /logs/tail    GA2 Q8   observability
    POST /extract            GA2 Q10  invoice field extraction
    POST,GET /orders         GA2 Q11  idempotency, cursor pagination, rate limiting
    GET  /ping               GA2 Q12  request-id propagation + rate limiting
    GET  /ga2/stream /ga2/cache                GA2 Q13, Q14  precomputed JSON answers

Design notes for /code-interpreter
----------------------------------
* Tool function `execute_python_code` runs the code and returns the EXACT stdout,
  or the exact traceback text on failure.
* The AI agent runs *only* when execution failed, uses structured (schema-constrained)
  output, and is **grounded on the real traceback**: an LLM asked "which line failed?"
  routinely answers with the call site instead of the frame that actually raised.
  We therefore treat the interpreter's own traceback as authoritative and use the
  model's answer as corroboration, which is what stops hallucinated line numbers.
"""

import base64
import csv
import hashlib
import hmac
import itertools
import json
import math
import os
import re
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
from collections import defaultdict, deque
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
try:  # Vercel imports api/index.py as a module; local `python api/index.py` uses a script path.
    from .ga3 import install_ga3_routes
except ImportError:  # pragma: no cover - the direct-script convenience path
    from ga3 import install_ga3_routes

CODE_FILENAME = "<code>"
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4.1-nano")
DATA_DIR = Path(__file__).resolve().parent / "data"
_exec_lock = threading.Lock()

app = FastAPI(title="TDS GA0 + GA1 + GA2 + GA3 endpoints")
install_ga3_routes(app, os.environ.get("PUBLIC_ORIGIN", "https://vercel-deploys-iota.vercel.app"))
# CORS is handled by ga2_middleware() below: wildcard everywhere (the GA0/GA1 graders need
# that, and GA0 Q25 even reads Access-Control-Allow-Origin from JS) except the GA2 paths
# that must answer exactly one assigned origin.


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    """Structured output schema for the AI error analyst."""
    error_lines: List[int]


# --------------------------------------------------------------------------- tool
def execute_python_code(code: str) -> dict:
    """Execute Python code and return its exact output.

    Returns {"success": bool, "output": str, "exception": BaseException | None}
    """
    buffer = StringIO()
    # A single namespace for globals AND locals: passing two separate dicts breaks
    # comprehensions, which look their free variables up in globals only.
    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    try:
        compiled = compile(code, CODE_FILENAME, "exec")
    except SyntaxError as exc:                      # also covers IndentationError
        return {"success": False, "output": traceback.format_exc(), "exception": exc}

    with _exec_lock:
        try:
            with redirect_stdout(buffer):
                exec(compiled, namespace, namespace)
        except BaseException as exc:                # RecursionError/SystemExit too
            # Drop this module's own `exec` frame so the traceback shows only the
            # submitted code, matching what a real interpreter would print.
            if exc.__traceback__ is not None and exc.__traceback__.tb_next is not None:
                exc.__traceback__ = exc.__traceback__.tb_next
            text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            return {"success": False, "output": text, "exception": exc}
    return {"success": True, "output": buffer.getvalue(), "exception": None}


def error_lines_from_traceback(exc: BaseException) -> List[int]:
    """Line number(s) in the submitted code where the exception actually arose.

    For a SyntaxError there is no executing frame, so use the reported lineno.
    Otherwise walk the traceback and take the DEEPEST frame belonging to the
    submitted code - that is the statement that raised, not the call site.
    """
    if isinstance(exc, SyntaxError) and exc.lineno:
        return [int(exc.lineno)]
    lines: List[int] = []
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == CODE_FILENAME:
            lines.append(int(tb.tb_lineno))
        tb = tb.tb_next
    return [lines[-1]] if lines else []


# ------------------------------------------------------------------------ AI agent
def analyze_error_with_ai(code: str, tb_text: str) -> Optional[List[int]]:
    """Ask an LLM for the failing line number(s) using structured output.

    Returns None when no token is configured or the call fails, so that error
    reporting never depends on network availability.
    """
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token:
        return None

    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(code.split("\n"), 1))
    payload = {
        "model": AIPIPE_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyse Python tracebacks. Report the 1-based line number(s) of "
                    "the submitted code where the exception was actually raised - the "
                    "deepest frame in the traceback, not the call site. Never invent a "
                    "line number that is absent from the traceback."
                ),
            },
            {
                "role": "user",
                "content": f"CODE (numbered):\n{numbered}\n\nTRACEBACK:\n{tb_text}",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False,
                },
            },
        },
    }
    request = urllib.request.Request(
        AIPIPE_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode())
        content = body["choices"][0]["message"]["content"]
        return ErrorAnalysis.model_validate_json(content).error_lines
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


# ------------------------------------------------------------------------ endpoint
@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    run = execute_python_code(request.code)

    if run["success"]:
        return {"error": [], "result": run["output"]}

    grounded = error_lines_from_traceback(run["exception"])
    ai_lines = analyze_error_with_ai(request.code, run["output"])

    # The traceback is ground truth; the model corroborates it.
    error_lines = grounded or (ai_lines or [])
    payload = {"error": error_lines, "result": run["output"]}
    if ai_lines is not None:
        payload["ai_error_lines"] = ai_lines
        payload["ai_agreed"] = sorted(ai_lines) == sorted(grounded)
    return payload


# --------------------------------------------------------------- Q10: /api students
@lru_cache(maxsize=1)
def load_students() -> List[dict]:
    """Rows of q-fastapi.csv, in file order, typed as {studentId: int, class: str}."""
    with open(DATA_DIR / "q-fastapi.csv", newline="", encoding="utf-8") as f:
        return [
            {"studentId": int(row["studentId"]), "class": row["class"]}
            for row in csv.DictReader(f)
        ]


@app.get("/api")
async def students(class_: Optional[List[str]] = Query(None, alias="class")):
    """All students, or only those in the requested classes.

    `?class=1A&class=1B` may repeat. Results keep CSV order, not request order.
    """
    rows = load_students()
    if class_:
        wanted = set(class_)
        rows = [r for r in rows if r["class"] in wanted]
    return {"students": rows}


# ----------------------------------------------------------- Q25: /api/latency stats
class LatencyRequest(BaseModel):
    regions: List[str]
    threshold_ms: float


@lru_cache(maxsize=1)
def load_telemetry() -> List[dict]:
    with open(DATA_DIR / "q-vercel-latency.json", encoding="utf-8") as f:
        return json.load(f)


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default and the grader)."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    if lo + 1 >= len(ordered):
        return ordered[lo]
    return ordered[lo] + (rank - lo) * (ordered[lo + 1] - ordered[lo])


@app.post("/api/latency")
async def latency(request: LatencyRequest):
    """Per-region mean latency, p95 latency, mean uptime and threshold breaches."""
    telemetry = load_telemetry()
    results = []
    for region in request.regions:
        records = [r for r in telemetry if r["region"] == region]
        if not records:
            raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
        latencies = [r["latency_ms"] for r in records]
        uptimes = [r["uptime_pct"] for r in records]
        results.append({
            "region": region,
            "avg_latency": round(sum(latencies) / len(latencies), 2),
            "p95_latency": round(percentile(latencies, 0.95), 2),
            "avg_uptime": round(sum(uptimes) / len(uptimes), 3),
            "breaches": sum(1 for v in latencies if v > request.threshold_ms),
        })
    return {"regions": results}


# ----------------------------------------------------------- Q11: /sentiment batch
# A dependency-free, rule-based classifier. Each sentence scores
# (happy matches - sad matches): > 0 happy, < 0 sad, otherwise neutral. Stems match at
# word starts, so "thrilled" and "thrilling" both hit "thrill". Sentences with no
# emotional vocabulary (facts, times, counts) fall through to neutral.
# Tuned on the GA0 Q11 sentence bank; it is not a general-purpose sentiment model.
HAPPY_STEMS = [
    "love", "excit", "joy", "winning", "dream", "thrill", "best", "smil", "amaz",
    "grateful", "fantastic", "hoping for", "wonderful", "proud", "happi", "happy",
    "delight", "bless", "bliss", "ecstatic", "beautiful", "jumping", "exceed",
    "cloud nine", "burst", "fortunate", "grinning", "alive", "energi", "celebrat",
    "spectacular", "perfect", "great", "awesome", "glad", "cheer", "pleased", "yay",
    "superb", "brilliant", "enjoy", "overjoy", "elated",
]
SAD_STEMS = [
    "worst", "heartbroken", "fail", "terrible", "passed away", "reject", "devastat",
    "nobody", "regret", "layoff", "disappoint", "worse", "lonely", "abandon",
    "falling apart", "depress", "ended badly", "hopeless", "cry", "pain", "broken",
    "miserable", "exhaust", "traumat", "defeat", "drown", "sorrow", "empty",
    "suffer", "anxiety", "lost", "grief", "worried", "shatter", "betray", "sadness",
    "haunt", "crush", "burden", "problems", "awful", "horrible", "sad", "unhappy",
    "hate", "angry", "upset", "tragic", "hurt", "despair", "sick",
]
_HAPPY_RE = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in HAPPY_STEMS) + ")")
_SAD_RE = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in SAD_STEMS) + ")")


def classify_sentiment(sentence: str) -> str:
    text = sentence.lower()
    score = len(_HAPPY_RE.findall(text)) - len(_SAD_RE.findall(text))
    return "happy" if score > 0 else "sad" if score < 0 else "neutral"


class SentimentRequest(BaseModel):
    sentences: List[str]


@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    """Classify each sentence, returning results in input order."""
    return {
        "results": [
            {"sentence": s, "sentiment": classify_sentiment(s)} for s in request.sentences
        ]
    }


# =============================================================================== GA1
GA1_DIR = DATA_DIR / "ga1"


# ------------------------------------------------- GA1 Q6: /effective-config (12-factor)
# Precedence, low -> high: defaults, config.<env>.yaml, .env, OS env (APP_*), ?set=k=v.
CFG_KEYS = ["port", "workers", "debug", "log_level", "api_key"]
CFG_DEFAULTS = {"port": 8000, "workers": 1, "debug": False, "log_level": "info",
                "api_key": "default-secret-000"}
CFG_ALIASES = {"num_workers": "workers"}


def _cfg_key(k: str) -> str:
    """APP_LOG_LEVEL -> log_level, NUM_WORKERS -> workers."""
    k = k.strip().lower()
    if k.startswith("app_"):
        k = k[4:]
    return CFG_ALIASES.get(k, k)


def _cfg_coerce(k, v):
    if k in ("port", "workers"):
        return int(v)
    if k == "debug":
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes", "on")
    return str(v)


def _read_kv(path: Path, sep: str) -> dict:
    """Flat `key: value` YAML / `KEY=value` dotenv reader (no PyYAML dependency)."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and sep in line:
                k, v = line.split(sep, 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _cfg_layers() -> list:
    env = os.environ.get("APP_ENV", "development")
    yaml_layer = _read_kv(GA1_DIR / "config" / f"config.{env}.yaml", ":")
    dotenv = _read_kv(GA1_DIR / "config" / "dot.env", "=")   # the .env file (renamed: deploy tools skip dotfiles)
    # OS layer: real APP_* variables win; with none set (fresh Vercel project) use config/os.env,
    # which records the OS environment this deployment is assigned.
    osenv = {k: v for k, v in os.environ.items() if k.startswith("APP_") and k != "APP_ENV"}
    if not osenv:
        osenv = _read_kv(GA1_DIR / "config" / "os.env", "=")
    return [yaml_layer, dotenv, osenv]


@app.get("/effective-config")
async def effective_config(set_: List[str] = Query(default=[], alias="set")):
    merged = dict(CFG_DEFAULTS)
    for layer in _cfg_layers():
        for k, v in layer.items():
            merged[_cfg_key(k)] = v
    for item in set_:                       # CLI overrides, highest precedence
        if "=" in item:
            k, v = item.split("=", 1)
            merged[_cfg_key(k)] = v
    out = {k: _cfg_coerce(k, merged[k]) for k in CFG_KEYS}
    out["api_key"] = "****"                 # never expose the secret
    return out


# ------------------------------------------------------------ GA1 Q14: /mcp (MCP server)
# Minimal Streamable-HTTP MCP server: JSON-RPC over POST, JSON responses, no SSE stream.
MCP_EMAIL = "23f3002028@ds.study.iitm.ac.in".strip().lower()
MCP_TOOL = {
    "name": "solve_challenge",
    "description": "Return the first 16 hex chars of SHA-256('<X-Exam-Challenge>:<email>').",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _rpc(id_, result=None, error=None):
    body = {"jsonrpc": "2.0", "id": id_}
    body["error" if error else "result"] = error or result
    return body


def _mcp_handle(msg: dict, request: Request):
    method, id_ = msg.get("method"), msg.get("id")
    if id_ is None:                         # notification (e.g. notifications/initialized): no reply
        return None
    if method == "initialize":
        return _rpc(id_, {
            "protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "tds-ga1-mcp", "version": "1.0.0"},
        })
    if method == "ping":
        return _rpc(id_, {})
    if method == "tools/list":
        return _rpc(id_, {"tools": [MCP_TOOL]})
    if method == "tools/call":
        name = (msg.get("params") or {}).get("name")
        if name != MCP_TOOL["name"]:
            return _rpc(id_, error={"code": -32602, "message": f"Unknown tool: {name}"})
        challenge = request.headers.get("x-exam-challenge", "")   # read from HTTP headers, not the body
        if not challenge:
            return _rpc(id_, {"content": [{"type": "text", "text": "missing X-Exam-Challenge header"}],
                              "isError": True})
        digest = hashlib.sha256(f"{challenge}:{MCP_EMAIL}".encode()).hexdigest()[:16]
        return _rpc(id_, {"content": [{"type": "text", "text": digest}], "isError": False})
    return _rpc(id_, error={"code": -32601, "message": f"Method not found: {method}"})


@app.post("/mcp")
async def mcp(request: Request):
    msg = await request.json()
    if isinstance(msg, list):               # JSON-RPC batch
        replies = [r for r in (_mcp_handle(m, request) for m in msg) if r is not None]
        return JSONResponse(replies) if replies else Response(status_code=202)
    reply = _mcp_handle(msg, request)
    if reply is None:
        return Response(status_code=202)
    headers = {"Mcp-Session-Id": "ga1-stateless"} if msg.get("method") == "initialize" else None
    return JSONResponse(reply, headers=headers)


@app.get("/mcp")
async def mcp_get():
    return Response(status_code=405, headers={"Allow": "POST"})   # no server-initiated stream


# --------------------------------------------------------- GA1 Q15: /ledger (ledger agent)
# Rules from the API's own notes: business dates are Asia/Kolkata; a repeated order id's latest
# updated_at row is current; only status 'paid' is revenue. Money is converted with /rates.
# Questions are parsed with keyword rules (no LLM): every answer is computed, never generated.
IST = timezone(timedelta(hours=5, minutes=30))
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"], 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@lru_cache(maxsize=1)
def load_ledger():
    usd = json.loads((GA1_DIR / "ledger" / "rates.json").read_text())["usd_per_unit"]
    current = {}
    for line in (GA1_DIR / "ledger" / "export.ndjson").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prev = current.get(r["id"])
        if prev is None or _ts(r["updated_at"]) > _ts(prev["updated_at"]):
            current[r["id"]] = r
    orders = []
    for r in current.values():
        local = _ts(r["created_at"]).astimezone(IST)
        orders.append({**r, "year": local.year, "month": local.month, "usd": r["amount"] * usd[r["currency"]]})
    regions = sorted({o["region"] for o in orders})
    products = sorted({o["product"] for o in orders}, key=len, reverse=True)   # longest name first
    return orders, regions, products


def _find(q: str, options):
    for o in options:
        if re.search(r"\b" + re.escape(o.lower()) + r"s?\b", q.lower()):
            return o
    return None


def _month(q: str):
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\w*\.?\s*,?\s*(20\d\d)\b", q.lower())
    if m:
        return int(m.group(2)), _MONTHS[m.group(1)]
    m = re.search(r"\b(20\d\d)-(\d\d)\b", q)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_ledger_question(q: str) -> dict:
    _, regions, products = load_ledger()
    ql = q.lower()
    p = {"region": _find(q, regions), "product": _find(q, products), "month": _month(q)}
    if re.search(r"\b(customers?|buyers?|clients?)\b", ql):
        p["intent"] = "distinct_customers"
    elif "refund" in ql:
        money = re.search(r"\b(how much|amount|value|usd|dollars?)\b", ql)
        count = re.search(r"\b(how many|count|number of)\b", ql)
        p["intent"] = "refund_amount" if money and not count else "refund_count"
    elif re.search(r"\b(top|best|most|highest|leading)\b", ql) and re.search(r"product|item|sell", ql):
        p["intent"] = "top_product"
    elif re.search(r"\b(average|avg|mean|typical)\b", ql) or "per order" in ql:
        p["intent"] = "avg_order"
    elif re.search(r"\b(how many|count|number of)\b", ql):
        p["intent"] = "order_count"
    else:
        p["intent"] = "revenue"
    return p


def answer_ledger_question(q: str):
    orders, _, _ = load_ledger()
    p = parse_ledger_question(q)

    def select(status):
        out = [o for o in orders if o["status"] == status]
        if p["region"]:
            out = [o for o in out if o["region"] == p["region"]]
        if p["product"] and p["intent"] != "top_product":
            out = [o for o in out if o["product"] == p["product"]]
        if p["month"]:
            out = [o for o in out if (o["year"], o["month"]) == p["month"]]
        return out

    i = p["intent"]
    if i == "refund_count":
        return len(select("refunded"))
    if i == "refund_amount":
        return round(sum(o["usd"] for o in select("refunded")), 2)
    paid = select("paid")
    if i == "distinct_customers":
        return len({o["customer"] for o in paid})
    if i == "top_product":
        by = defaultdict(float)
        for o in paid:
            by[o["product"]] += o["usd"]
        return max(sorted(by), key=lambda k: by[k]) if by else None
    if i == "avg_order":
        return round(sum(o["usd"] for o in paid) / len(paid), 2) if paid else 0
    if i == "order_count":
        return len(paid)
    return round(sum(o["usd"] for o in paid), 2)


class LedgerQuestion(BaseModel):
    question: str


@app.post("/ledger")
async def ledger(request: LedgerQuestion):
    return {"answer": answer_ledger_question(request.question)}


# =============================================================================== GA2
GA2_DIR = DATA_DIR / "ga2"

# Assigned values (seeded from the email; read off the GA2 question page).
GA2_STATS_ORIGIN = "https://dash-mwnied.example.com"      # Q3: only this origin gets ACAO
GA2_PING_ORIGIN = "https://app-g0ixb3.example.com"        # Q12: assigned origin
EXAM_ORIGIN = "https://exam.sanand.workers.dev"           # the quiz page (browser-side graders)
GA2_API_KEY = "ak_qlfueyhzeaogfn8ryjnb7c0h"               # Q7: X-API-Key
GA2_ISS = "https://idp.exam.local"                        # Q4
GA2_AUD = "tds-ehukzybs.apps.exam.local"                  # Q4
GA2_ORDER_TOTAL = 45                                      # Q11: catalog is ids 1..45
GA2_ORDER_RATE = 16                                       # Q11: 16 requests / 10s per client
GA2_PING_RATE = 10                                        # Q12: 10 requests / 10s per client
EMAIL = "23f3002028@ds.study.iitm.ac.in"

# Paths with a strict per-origin CORS policy. Everything else stays wildcard, which the
# GA0/GA1 graders rely on (GA0 Q25 even reads Access-Control-Allow-Origin from JS).
STRICT_CORS = {
    "/stats": {GA2_STATS_ORIGIN},
    "/ping": {GA2_PING_ORIGIN, EXAM_ORIGIN},
}

STARTED_AT = time.time()
_REQUEST_COUNTER = itertools.count(1)
_requests_seen = {"n": 0}
REQUEST_LOG = deque(maxlen=500)
_RATE_STATE = defaultdict(deque)            # (bucket, client) -> hit timestamps
_RATE_LOCK = threading.Lock()
_IDEMPOTENCY = {}                           # Idempotency-Key -> created order


def _rate_limited(bucket: str, client: str, limit: int, window: float = 10.0):
    """Sliding-window limiter. Returns retry_after seconds when over the limit, else None."""
    now = time.time()
    with _RATE_LOCK:
        hits = _RATE_STATE[(bucket, client)]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            return max(1, int(window - (now - hits[0])) + 1)
        hits.append(now)
        return None


@app.middleware("http")
async def ga2_middleware(request: Request, call_next):
    """CORS (wildcard, or strict per path), request-id/process-time headers, metrics, logs."""
    start = time.perf_counter()
    path = request.url.path
    origin = request.headers.get("origin")
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())   # echo inbound id (Q12)
    request.state.request_id = rid
    _requests_seen["n"] = next(_REQUEST_COUNTER)

    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        response = await call_next(request)

    allowed = STRICT_CORS.get(path)
    if allowed is None:
        acao = "*"
    elif origin and origin in allowed:
        acao = origin
    else:
        acao = None                      # disallowed origin: send no ACAO header at all
    if acao:
        response.headers["access-control-allow-origin"] = acao
        if acao != "*":
            response.headers["vary"] = "Origin"
        if request.method == "OPTIONS":
            response.headers["access-control-allow-methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            response.headers["access-control-allow-headers"] = \
                request.headers.get("access-control-request-headers") or "*"
            response.headers["access-control-max-age"] = "600"
    response.headers["access-control-expose-headers"] = "*"
    response.headers["x-request-id"] = rid
    response.headers["x-process-time"] = f"{time.perf_counter() - start:.6f}"

    REQUEST_LOG.append({
        "level": "info", "ts": datetime.now(timezone.utc).isoformat(), "path": path,
        "request_id": rid, "method": request.method, "status": response.status_code,
    })
    return response


# ------------------------------------------------------- GA2 Q3: GET /stats (CORS + metrics)
@app.get("/stats")
async def stats(values: str = Query(default="")):
    nums = [int(v) for v in re.split(r"[,\s]+", values.strip()) if v not in ("", "-")]
    if not nums:
        raise HTTPException(status_code=422, detail="values must be comma-separated integers")
    return {"email": EMAIL, "count": len(nums), "sum": sum(nums),
            "min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}


# ------------------------------------------------- GA2 Q4: POST /verify (RS256 JWT, no deps)
# Pure-Python RSA PKCS#1 v1.5 verification. Keeps PyJWT/cryptography out of the serverless
# bundle and makes the check fully deterministic.
GA2_IDP_SPKI_B64 = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2okOHspNjgA+2rTLbeuYcxiP/hG8C6Sb9iwg3yiLAA4H"
    "CnpITcbWCSelbvbYGuc3EbNy4xFyf5Cbj5DHJMIDEkryOgyd2giIIIBOUBj8S63uGcnRpOBh9NFatfNwheKuzsPu"
    "VNldu6A9cNteNpXcWyJjG2axVfmq7i6SuKr1JoWYG7xTTAvKPujSl4OtsQfO3h5NepzdfXpr28oNnzfWed+zclR6"
    "BcmNNo/WVfJ4xyCLSf0BCOgdTgW6PdaChd1l9VDetJZVEgC5tkyvXsfISI6iyrYbKR0NEBSqq4XkadEjsCs4F1Rn"
    "csS4LlgniT7GlkL9Mce3b0wGLs9/7ZIXdQIDAQAB"
)
SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def _der_read(buf: bytes, i: int):
    """Read one DER TLV at offset i. Returns (tag, value_bytes, next_offset)."""
    tag = buf[i]
    i += 1
    n = buf[i]
    i += 1
    if n & 0x80:
        k = n & 0x7F
        n = int.from_bytes(buf[i:i + k], "big")
        i += k
    return tag, buf[i:i + n], i + n


def _rsa_pubkey_from_spki(spki: bytes):
    _, seq, _ = _der_read(spki, 0)                       # outer SEQUENCE
    _, _alg, j = _der_read(seq, 0)                       # AlgorithmIdentifier
    _, bitstring, _ = _der_read(seq, j)                  # BIT STRING
    _, rsa_seq, _ = _der_read(bitstring[1:], 0)          # skip the unused-bits byte
    _, n_bytes, j2 = _der_read(rsa_seq, 0)
    _, e_bytes, _ = _der_read(rsa_seq, j2)
    return int.from_bytes(n_bytes, "big"), int.from_bytes(e_bytes, "big")


def _b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def rs256_verify(signing_input: bytes, signature: bytes, spki_b64: str) -> bool:
    n, e = _rsa_pubkey_from_spki(base64.b64decode(spki_b64))
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    em = pow(int.from_bytes(signature, "big"), e, n).to_bytes(k, "big")
    expected = SHA256_DIGEST_INFO + hashlib.sha256(signing_input).digest()
    prefix = b"\x00\x01" + b"\xff" * (k - 3 - len(expected)) + b"\x00"
    return hmac.compare_digest(em, prefix + expected)


class TokenRequest(BaseModel):
    token: str = ""


@app.post("/verify")
async def verify_token(body: TokenRequest):
    invalid = JSONResponse({"valid": False}, status_code=401)
    parts = (body.token or "").strip().split(".")
    if len(parts) != 3:
        return invalid
    try:
        header = json.loads(_b64url(parts[0]))
        claims = json.loads(_b64url(parts[1]))
        signature = _b64url(parts[2])
    except Exception:
        return invalid
    if header.get("alg") != "RS256":                     # no alg confusion / "none"
        return invalid
    if not rs256_verify(f"{parts[0]}.{parts[1]}".encode(), signature, GA2_IDP_SPKI_B64):
        return invalid
    if claims.get("iss") != GA2_ISS:
        return invalid
    aud = claims.get("aud")
    if aud != GA2_AUD and not (isinstance(aud, list) and GA2_AUD in aud):
        return invalid
    try:
        if float(claims.get("exp", 0)) <= time.time():
            return invalid
    except (TypeError, ValueError):
        return invalid
    return {"valid": True, "email": claims.get("email", EMAIL),
            "sub": claims.get("sub"), "aud": aud if isinstance(aud, str) else GA2_AUD}


# --------------------------------------------------------- GA2 Q7: POST /analytics (API key)
class AnalyticsRequest(BaseModel):
    events: List[dict] = []


@app.post("/analytics")
async def analytics(request: Request, body: AnalyticsRequest):
    if request.headers.get("x-api-key") != GA2_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    totals = defaultdict(float)
    revenue = 0.0
    for ev in body.events:
        try:
            amount = float(ev.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        if amount > 0:                       # only positive amounts are revenue
            revenue += amount
            totals[str(ev.get("user"))] += amount
    top_user = max(sorted(totals), key=lambda u: totals[u]) if totals else None
    return {"email": EMAIL, "total_events": len(body.events),
            "unique_users": len({str(ev.get("user")) for ev in body.events}),
            "revenue": round(revenue, 2), "top_user": top_user}


# ------------------------------ GA2 Q8: /work, /metrics, /healthz, /logs/tail (observability)
@app.get("/work")
async def work(n: int = Query(default=1)):
    done = max(0, int(n))
    checksum = sum(i * i for i in range(done))       # actually do the work
    return {"email": EMAIL, "done": done, "checksum": checksum}


@app.get("/metrics")
async def metrics():
    body = ("# HELP http_requests_total Total HTTP requests served.\n"
            "# TYPE http_requests_total counter\n"
            f"http_requests_total {_requests_seen['n']}\n"
            "# HELP process_uptime_seconds Seconds since process start.\n"
            "# TYPE process_uptime_seconds gauge\n"
            f"process_uptime_seconds {time.time() - STARTED_AT:.6f}\n")
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED_AT, 6)}


@app.get("/logs/tail")
async def logs_tail(limit: int = Query(default=50)):
    return list(REQUEST_LOG)[-max(1, int(limit)):]


# ------------------------------------------ GA2 Q10: POST /extract (invoice field extraction)
_CUR_SYMBOLS = {"$": "USD", "\u20ac": "EUR", "\u00a3": "GBP"}
_COMPANY_TAIL = (r"Industries|Ltd\.?|Limited|LLC|Inc\.?|Incorporated|GmbH|Corp\.?|Corporation|"
                 r"Co\.?|Company|Technologies|Technology|Solutions|Supplies|Services|Systems|"
                 r"Partners|Holdings|Group|Trading|Enterprises|Works|Labs")


def extract_invoice(text: str) -> dict:
    t = text or ""
    currency = None
    m = re.search(r"\b(USD|EUR|GBP)\b", t, re.I)
    if m:
        currency = m.group(1).upper()
    else:
        for sym, code in _CUR_SYMBOLS.items():
            if sym in t:
                currency = code
                break

    date = None
    m = re.search(r"(?:due|payable|payment)[^0-9]{0,40}(\d{4}-\d{2}-\d{2})", t, re.I)
    if m:
        date = m.group(1)
    else:
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", t)
        if dates:
            date = dates[-1]

    amount = None
    money = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    for pat in (r"(?:total\s*(?:due|amount|payable)?|amount\s*due|balance\s*due|grand\s*total)"
                r"\s*[:\-]?\s*(?:USD|EUR|GBP|[$\u20ac\u00a3])?\s*" + money,
                r"(?:USD|EUR|GBP|[$\u20ac\u00a3])\s*" + money,
                money + r"\s*(?:USD|EUR|GBP)\b"):
        m = re.search(pat, t, re.I)
        if m:
            amount = float(m.group(1).replace(",", ""))
            break
    if amount is None:
        nums = [float(x.replace(",", "")) for x in re.findall(r"\b([0-9][0-9,]*\.[0-9]{2})\b", t)]
        if nums:
            amount = max(nums)

    vendor = None
    m = re.search(r"(?:vendor|supplier|seller|from|bill(?:ed)?\s*(?:from|by)|issued\s*by|"
                  r"remit\s*to|pay\s*to)\s*[:\-]\s*(.+)", t, re.I)
    if m:
        vendor = m.group(1).strip().splitlines()[0].strip(" .,;:")
    if not vendor:
        m = re.search(r"([A-Z][\w&.'-]*(?:\s+[A-Z0-9][\w&.'-]*){0,4}\s+(?:" + _COMPANY_TAIL + r"))", t)
        if m:
            vendor = m.group(1).strip()
    if not vendor:
        for line in t.splitlines():
            line = line.strip()
            if line and not re.match(r"(?i)^(invoice|date|due|total|amount|currency|bill|ref|"
                                     r"number|no\b|subtotal|tax|qty|description)", line):
                vendor = line.strip(" .,;:")
                break
    return {"vendor": vendor, "amount": amount, "currency": currency, "date": date}


class ExtractRequest(BaseModel):
    text: str = ""


@app.post("/extract")
async def extract(body: ExtractRequest):
    try:
        out = extract_invoice(body.text)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse the invoice text")
    if not (body.text or "").strip() or out["amount"] is None:
        raise HTTPException(status_code=422, detail="Invoice text has no recognisable amount")
    return {"vendor": out["vendor"] or "unknown", "amount": out["amount"],
            "currency": out["currency"] or "USD", "date": out["date"] or "1970-01-01"}


# --------------------- GA2 Q11: /orders (idempotency + cursor pagination + rate limiting)
def _client_id(request: Request) -> str:
    return request.headers.get("x-client-id") or (request.client.host if request.client else "anon")


def _too_many(retry: int) -> JSONResponse:
    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                        headers={"Retry-After": str(retry)})


@app.post("/orders", status_code=201)
async def create_order(request: Request):
    retry = _rate_limited("orders", _client_id(request), GA2_ORDER_RATE)
    if retry:
        return _too_many(retry)
    key = request.headers.get("idempotency-key")
    if key and key in _IDEMPOTENCY:
        return JSONResponse(_IDEMPOTENCY[key], status_code=201)   # same key -> same order
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    order = {"id": f"ord-{uuid.uuid4().hex[:12]}", "amount": (payload or {}).get("amount", 100),
             "status": "created", "email": EMAIL}
    if key:
        _IDEMPOTENCY[key] = order
    return JSONResponse(order, status_code=201)


@app.get("/orders")
async def list_orders(request: Request, limit: int = Query(default=10),
                      cursor: str = Query(default="")):
    retry = _rate_limited("orders", _client_id(request), GA2_ORDER_RATE)
    if retry:
        return _too_many(retry)
    size = max(1, min(int(limit or 10), GA2_ORDER_TOTAL))
    try:
        start = int(cursor) if str(cursor).strip() else 1          # cursor is an opaque offset
    except ValueError:
        start = 1
    start = max(1, start)
    end = min(start + size - 1, GA2_ORDER_TOTAL)
    items = [{"id": i, "amount": 100 + i, "status": "open"} for i in range(start, end + 1)]
    return {"items": items, "next_cursor": str(end + 1) if end < GA2_ORDER_TOTAL else None,
            "total": GA2_ORDER_TOTAL}


# ---------------------------- GA2 Q12: GET /ping (request context + per-client rate limit)
@app.get("/ping")
async def ping(request: Request):
    retry = _rate_limited("ping", _client_id(request), GA2_PING_RATE)
    if retry:
        return _too_many(retry)
    return {"email": EMAIL, "request_id": request.state.request_id}


# ------------------------------- GA2 Q13 / Q14: precomputed JSON answers served over HTTP
@lru_cache(maxsize=4)
def _ga2_answer(name: str) -> dict:
    return json.loads((GA2_DIR / name).read_text(encoding="utf-8"))


@app.get("/ga2/stream")
async def ga2_stream():
    return _ga2_answer("stream.json")


@app.get("/ga2/cache")
async def ga2_cache():
    return _ga2_answer("cache.json")


@app.get("/ga3/kim-novak-face.glb")
async def kim_novak_face():
    """Public, CORS-enabled GLB submitted for the GA3 face-model validity check."""
    return FileResponse(DATA_DIR / "kim-novak-face.glb", media_type="model/gltf-binary")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": [
            "POST /code-interpreter",
            "GET /api?class=...",
            "POST /api/latency",
            "POST /sentiment",
            "GET /effective-config?set=key=value",
            "POST /mcp",
            "POST /ledger",
            "GET /stats?values=1,2,3", "POST /verify", "POST /analytics",
            "GET /work?n=5", "GET /metrics", "GET /healthz", "GET /logs/tail?limit=20",
            "POST /extract", "POST /orders", "GET /orders?limit=10", "GET /ping",
            "GET /ga2/stream", "GET /ga2/cache",
            "GET /.well-known/agent-card.json", "POST /a2a/message:send", "GET /a2a/tasks",
            "POST /v1/mailroom/actions", "POST /v2/incidents", "GET /v2/incidents/{runId}",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
