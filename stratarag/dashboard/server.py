"""stratarag Playground: a zero-dependency local dev UI.

    import stratarag as mn
    from stratarag.dashboard import serve

    agent = mn.Agent(model="echo", knowledge=kb, memory=mn.Memory(episodic=True))
    serve(agent, port=7327)   # open http://localhost:7327

Chat with the agent and watch the "recall strip": which facts it remembered,
which sources it retrieved, the stage-by-stage trace, and the confidence gate.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..agent import Agent
from .page import PAGE_HTML


def _result_json(result) -> dict:
    return {
        "answer": result.output,
        "confidence": round(result.confidence, 3),
        "gated": result.gated,
        "sources": [
            {"text": sc.chunk.text, "score": round(sc.score, 3),
             "section": (sc.chunk.metadata or {}).get("section", "")}
            for sc in result.sources
        ],
        "memory": {
            kind: [r.content for r in records]
            for kind, records in result.memory_used.items()
        },
        "trace": [
            {"stage": t.stage, "ms": t.elapsed_ms, "detail": t.detail}
            for t in result.trace
        ],
    }


def make_handler(agent: Agent):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default request logging
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/health"):
                self._json(200, {"ok": True, "tools": len(agent.tools),
                                 "has_memory": agent.memory is not None,
                                 "has_knowledge": agent.knowledge is not None})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid JSON body"})
            if self.path == "/api/chat":
                message = (body.get("message") or "").strip()
                if not message:
                    return self._json(400, {"error": "message is required"})
                user_id = body.get("user_id") or "default"
                try:
                    result = agent.run(message, user_id=user_id)
                except Exception as e:
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return self._json(200, _result_json(result))
            if self.path == "/api/remember":
                fact = (body.get("fact") or "").strip()
                if not fact or agent.memory is None or agent.memory.semantic is None:
                    return self._json(400, {"error": "fact required and semantic "
                                                     "memory must be enabled"})
                agent.memory.remember(fact, user_id=body.get("user_id") or "default")
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "not found"})

    return Handler


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 7327,
          block: bool = True) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(agent))
    if block:  # pragma: no cover - interactive
        print(f"stratarag playground -> http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
