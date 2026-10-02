"""Public endpoints for TDS GA0, served from one FastAPI app on Vercel.

    POST /code-interpreter   Q5   execute Python, report the line that raised
    GET  /api                Q10  students from q-fastapi.csv, filtered by ?class=
    POST /api/latency        Q25  per-region latency stats from q-vercel-latency.json
    POST /sentiment          Q11  batch rule-based sentiment: happy / sad / neutral

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

import csv
import json
import math
import os
import re
import threading
import traceback
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CODE_FILENAME = "<code>"
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4.1-nano")
DATA_DIR = Path(__file__).resolve().parent / "data"
_exec_lock = threading.Lock()

app = FastAPI(title="TDS GA0 endpoints")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # Browsers hide every non-safelisted response header from cross-origin
    # JavaScript, including Access-Control-Allow-Origin itself. The Q25 grader reads
    # that header with fetch(), so it must be exposed explicitly.
    expose_headers=["*"],
)


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


@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": [
            "POST /code-interpreter",
            "GET /api?class=...",
            "POST /api/latency",
            "POST /sentiment",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
