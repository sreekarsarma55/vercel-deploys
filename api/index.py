"""POST /code-interpreter - execute Python, and analyse errors to find line numbers.

Design notes
------------
* Tool function `execute_python_code` runs the code and returns the EXACT stdout,
  or the exact traceback text on failure.
* The AI agent runs *only* when execution failed, uses structured (schema-constrained)
  output, and is **grounded on the real traceback**: an LLM asked "which line failed?"
  routinely answers with the call site instead of the frame that actually raised.
  We therefore treat the interpreter's own traceback as authoritative and use the
  model's answer as corroboration, which is what stops hallucinated line numbers.
"""

import json
import os
import sys
import threading
import traceback
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CODE_FILENAME = "<code>"
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4.1-nano")
_exec_lock = threading.Lock()

app = FastAPI(title="Code Interpreter with AI Error Analysis")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "POST /code-interpreter"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
