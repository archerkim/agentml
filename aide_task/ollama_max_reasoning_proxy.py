#!/usr/bin/env python3
"""Local OpenAI-compatible proxy that enforces Ollama's maximum reasoning effort.

This is an infrastructure sidecar, not part of the AIDE workflow.  AIDE 0.2.2
does not expose arbitrary per-request model options, while Ollama's OpenAI
compatibility endpoint accepts ``reasoning_effort: \"max\"``.  The proxy adds that
field before forwarding every chat-completions request to the local Ollama server.

Run in a separate terminal before starting AIDE:
    python3 ollama_max_reasoning_proxy.py
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


LISTEN_HOST = os.environ.get("OLLAMA_REASONING_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("OLLAMA_REASONING_PROXY_PORT", "11435"))
UPSTREAM = urlsplit(os.environ.get("OLLAMA_UPSTREAM_URL", "http://127.0.0.1:11434"))

# One maximum-effort generation on a local 27B model routinely runs well past ten
# minutes: the reasoning trace alone can be tens of thousands of tokens.  The old
# hard-wired 600s upstream timeout cut those generations off mid-flight, and the
# resulting 502 reached AIDE as an empty completion -> `extract_code()` returned ""
# -> a zero-length candidate marked buggy (observed on logs/5-kuairand-pure-run1,
# steps 1 and 2).  Keep this at least as large as LLM_REQUEST_TIMEOUT_SEC in
# run_with_early_stop.py so the OpenAI client, not the proxy, owns the deadline.
UPSTREAM_TIMEOUT_SEC = float(os.environ.get("OLLAMA_UPSTREAM_TIMEOUT_SEC", "5400"))

# Reasoning depth applied to every chat-completions request.  "max" is the default
# and what this proxy was built for, but it is the single biggest cost driver: a
# draft costs 35-60K completion tokens and 12-21 minutes, and one run-6 draft spent
# 122885 tokens reasoning without ever emitting an answer.  Lowering this to
# "medium" trades reasoning depth for throughput and is the fallback if maximum
# effort proves unstable.  Override per process:
#     OLLAMA_REASONING_EFFORT=medium python3 ollama_max_reasoning_proxy.py
REASONING_EFFORT = os.environ.get("OLLAMA_REASONING_EFFORT", "max")

if UPSTREAM.scheme != "http" or not UPSTREAM.hostname:
    raise SystemExit("OLLAMA_UPSTREAM_URL must be an http:// URL with a host")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# A reasoning trace is full of half-written fragments ("we could do `df.groupby(...)`"),
# and AIDE's extract_code() concatenates EVERY fenced block that happens to compile.
# Handing it raw reasoning would therefore assemble a Frankenstein script out of
# unrelated snippets - measurably worse than an empty candidate, because it looks
# real enough to run.  Salvage only a single block that is both syntactically valid
# and substantial enough to plausibly be the actual solution.
MIN_SALVAGEABLE_CODE_CHARS = 600

# "Compiles" is far too weak a test on its own: a draft that trails off into
# `HIST_COLS = [...]` followed by a bare `...` is syntactically perfect Python and
# does nothing (observed exactly this on the first max-reasoning probe).  Require
# positive evidence that the model reached the END of its program - AIDE's
# implementation guideline mandates a submission.csv write for every task, so a
# block that contains one is a program the model finished rather than a fragment
# it was still sketching.
SALVAGE_COMPLETION_MARKER = "submission.csv"


def _salvage_code_from_reasoning(reasoning: str) -> str | None:
    """Return the largest complete, compilable program in a reasoning trace, or None."""
    import re

    candidates = []
    for match in re.finditer(r"```(?:python)?\n(.*?)```", reasoning, re.DOTALL):
        block = match.group(1).strip()
        if len(block) < MIN_SALVAGEABLE_CODE_CHARS:
            continue
        if SALVAGE_COMPLETION_MARKER not in block:
            continue
        try:
            compile(block, "<salvage>", "exec")
        except (SyntaxError, ValueError):
            continue
        candidates.append(block)

    return max(candidates, key=len) if candidates else None


class OllamaReasoningProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        _log(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        if self.path == "/_health":
            self._send(200, b'{"status":"ok"}', "application/json")
            return
        self._forward()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        if self.path == "/v1/chat/completions":
            try:
                request = json.loads(body)
            except json.JSONDecodeError:
                self._send(400, b'{"error":"invalid JSON"}', "application/json")
                return

            # AIDE 0.2.2 sends its structured execution-review prompt as a
            # system-only conversation. Qwen's OpenAI-compatible endpoint
            # rejects that shape (HTTP 500), whereas the semantically identical
            # single user message is supported. Its ordinary draft/improve calls
            # already contain user messages and are intentionally left alone.
            messages = request.get("messages")
            if (
                isinstance(messages, list)
                and len(messages) == 1
                and messages[0].get("role") == "system"
            ):
                request["messages"] = [
                    {"role": "user", "content": messages[0].get("content", "")}
                ]

            request["reasoning_effort"] = REASONING_EFFORT
            body = json.dumps(request, separators=(",", ":")).encode()
            prompt_chars = sum(
                len(m.get("content") or "") for m in request.get("messages") or []
            )
            _log(
                f"-> chat/completions model={request.get('model')} "
                f"prompt_chars={prompt_chars} (reasoning_effort={REASONING_EFFORT})"
            )

        self._forward(body)

    def _forward(self, body: bytes | None = None) -> None:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"connection", "content-length", "host", "transfer-encoding"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))

        upstream_path = f"{UPSTREAM.path.rstrip('/')}{self.path}"
        connection = http.client.HTTPConnection(
            UPSTREAM.hostname, UPSTREAM.port or 80, timeout=UPSTREAM_TIMEOUT_SEC
        )
        t0 = time.time()
        try:
            connection.request(self.command, upstream_path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
        except OSError as exc:
            _log(f"<- upstream error after {time.time() - t0:.0f}s: {exc}")
            self._send(502, json.dumps({"error": str(exc)}).encode(), "application/json")
            connection.close()
            return

        try:
            if self.path == "/v1/chat/completions" and response.status == 200:
                response_body = self._postprocess_completion(response_body, time.time() - t0)

            response_headers = {
                key: value
                for key, value in response.getheaders()
                if key.lower() not in {"connection", "content-length", "transfer-encoding"}
            }
            self.send_response(response.status, response.reason)
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (BrokenPipeError, ConnectionResetError):
            # The client (AIDE's OpenAI SDK) gave up before we answered - nothing
            # left to write to.  Log it instead of dumping a traceback that looks
            # like a proxy fault.
            _log("<- client disconnected before the response could be delivered")
        finally:
            connection.close()

    def _postprocess_completion(self, response_body: bytes, elapsed: float) -> bytes:
        """Log what actually came back, and salvage a reasoning-only response.

        Qwen at maximum reasoning effort sometimes spends its whole generation
        budget inside the reasoning channel and returns an EMPTY ``content``.
        AIDE only ever looks at ``content``, so such a response becomes a
        zero-length candidate.  When that happens and the reasoning text does
        contain a fenced code block, hand the reasoning over as the content so
        ``extract_code()`` has something real to parse - a partially-drafted
        solution is worth far more to the search than an empty node.
        """
        try:
            payload = json.loads(response_body)
            choice = payload["choices"][0]
            message = choice["message"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            _log(f"<- {elapsed:.0f}s (non-standard completion payload, passed through)")
            return response_body

        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        usage = payload.get("usage") or {}
        finish = choice.get("finish_reason")
        _log(
            f"<- {elapsed:.0f}s finish={finish} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"reasoning_chars={len(reasoning)} content_chars={len(content)}"
        )

        if finish == "length":
            _log("   WARNING: generation hit the context/length limit - "
                 "raise num_ctx on the model alias if this recurs")

        # A function-calling response (AIDE's execution review) legitimately has an
        # empty `content` - its answer lives in `tool_calls`, which the backend reads
        # first.  Salvage must not treat that as a truncated generation.
        if message.get("tool_calls"):
            return response_body

        if not content.strip() and reasoning.strip():
            salvaged = _salvage_code_from_reasoning(reasoning)
            if salvaged is None:
                _log("   SALVAGE: content was empty and the reasoning holds no complete "
                     "program - passing the empty response through so AIDE retries")
                return response_body
            _log(f"   SALVAGE: content was empty; promoting the largest complete program "
                 f"found in the reasoning ({len(salvaged)} chars)")
            message["content"] = (
                "Recovered from an interrupted reasoning trace: the model ran out of "
                "context before emitting its final answer, so this is the most complete "
                "program it had written while thinking.\n\n"
                f"```python\n{salvaged}\n```"
            )
            return json.dumps(payload, separators=(",", ":")).encode()

        return response_body

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            _log("<- client disconnected before the error response could be delivered")


if __name__ == "__main__":
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), OllamaReasoningProxy)
    print(
        f"Ollama maximum-reasoning proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT}/v1 "
        f"-> {UPSTREAM.geturl()} (reasoning_effort={REASONING_EFFORT}, "
        f"upstream timeout {UPSTREAM_TIMEOUT_SEC:.0f}s)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
